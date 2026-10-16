"""Additive modules around the uploaded CUT ResnetGenerator.

PatchNCE feature requests use the unconditioned backbone. Posterior logits are
computed before FiLM, so paired IHC/teacher conditions cannot leak into them.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.networks import ResnetBlock


class ConditionalGenerator(nn.Module):
    def __init__(self, backbone, channels, window=128, bins=5,learn_bin_means=False):
        super().__init__()
        self.backbone, self.window, self.bins = backbone, window, bins
        self.first_residual = next(i for i,m in enumerate(backbone.model) if isinstance(m, ResnetBlock))
        self.distribution_head = nn.Conv2d(channels, bins, 1)
        self.within_bin_head=nn.Conv2d(channels,bins,1) if learn_bin_means else None
        if self.within_bin_head is not None:
            nn.init.zeros_(self.within_bin_head.weight);nn.init.zeros_(self.within_bin_head.bias)
        self.pixel_head = nn.Conv2d(channels, 1, 1)
        self.film = nn.Sequential(nn.Conv2d(1, channels, 1), nn.ReLU(), nn.Conv2d(channels, 2*channels, 1))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        # A bottleneck-only affine signal can be largely erased by decoder
        # InstanceNorm. This second FiLM is after the final normalization, before
        # RGB projection; both paths start as exact identities.
        decoder_channels=channels//4
        self.decoder_film=nn.Sequential(nn.Conv2d(1,decoder_channels,1),nn.ReLU(),
                                         nn.Conv2d(decoder_channels,2*decoder_channels,1))
        nn.init.zeros_(self.decoder_film[-1].weight);nn.init.zeros_(self.decoder_film[-1].bias)

    def forward(self, image, layers=None, encode_only=False, condition=None,
                return_aux=False, sample=False, generator=None,teacher_mask=None):
        if layers:
            return self.backbone(image, list(layers), encode_only)
        if encode_only: raise ValueError('encode_only requires feature layer indices.')
        if any(s % self.window for s in image.shape[-2:]):
            raise ValueError('Image and statistic window dimensions are inconsistent.')
        feat = image
        for layer in self.backbone.model[:self.first_residual]: feat = layer(feat)
        unconditioned = feat
        grid = tuple(s//self.window for s in image.shape[-2:])
        pooled=F.adaptive_avg_pool2d(feat,grid)
        logits = self.distribution_head(pooled)
        probabilities = logits.softmax(1)
        centers = (torch.arange(self.bins, device=image.device, dtype=image.dtype)+.5)/self.bins
        lower=torch.arange(self.bins,device=image.device,dtype=image.dtype)[None,:,None,None]/self.bins
        bin_means=lower+self.within_bin_head(pooled).sigmoid()/self.bins if self.within_bin_head is not None else centers[None,:,None,None].expand_as(probabilities)
        expected=(probabilities*bin_means).sum(1,keepdim=True)
        if condition is None:
            if sample:
                flat = probabilities.permute(0,2,3,1).reshape(-1,self.bins)
                bins = torch.multinomial(flat, 1, generator=generator).view(image.shape[0],1,*grid)
                condition = bin_means.gather(1,bins)
            else:
                condition = expected
        elif teacher_mask is not None:
            if teacher_mask.shape!=condition.shape:raise ValueError('Teacher mask shape differs from the condition grid.')
            condition=torch.where(teacher_mask,condition,expected.detach())
        if condition.shape != (image.shape[0],1,*grid):
            raise ValueError('Condition must be B x 1 x statistic-grid-height x statistic-grid-width.')
        if not torch.isfinite(condition).all() or (condition<0).any() or (condition>1).any():
            raise ValueError('Conditions must be finite values in [0,1].')
        gamma,beta = self.film(F.interpolate(condition, size=feat.shape[-2:], mode='nearest')).chunk(2,1)
        feat = feat*(1+.5*torch.tanh(gamma)) + beta
        output = feat
        for index in range(self.first_residual,len(self.backbone.model)):
            if index==len(self.backbone.model)-3:
                dg,db=self.decoder_film(F.interpolate(condition,size=output.shape[-2:],mode='nearest')).chunk(2,1)
                output=output*(1+.5*torch.tanh(dg))+db
            output=self.backbone.model[index](output)
        pixel_logits = F.interpolate(self.pixel_head(unconditioned),
                                     size=image.shape[-2:], mode='bilinear', align_corners=False) if return_aux else None
        if return_aux:
            return output, dict(logits=logits, probabilities=probabilities, condition=condition,
                                pixel_logits=pixel_logits,bin_means=bin_means,expected_condition=expected)
        return output

    def unconditioned_features(self, image):
        feat=image
        for layer in self.backbone.model[:self.first_residual]: feat=layer(feat)
        return feat


class ErrorPredictor(nn.Module):
    """Deployable raw error score: H&E, generated RGB, sample-variance map only."""
    def __init__(self, window=128, width=16):
        super().__init__(); self.window=window
        self.net=nn.Sequential(nn.Conv2d(7,width,5,stride=4,padding=2), nn.ReLU(),
                               nn.Conv2d(width,width*2,3,stride=2,padding=1),nn.ReLU(),
                               nn.Conv2d(width*2,1,1))

    def forward(self, he, generated, variance):
        if he.shape!=generated.shape or variance.shape!=(he.shape[0],1,*he.shape[-2:]):
            raise ValueError('Risk inputs have incompatible dimensions.')
        grid=tuple(s//self.window for s in he.shape[-2:])
        return F.adaptive_avg_pool2d(self.net(torch.cat([he,generated,variance],1)),grid)
