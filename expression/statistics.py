"""Fixed stain measurement. Images are NCHW in [-1, 1].

The same H&E-derived foreground is used for reference and generated statistics.
The IHC foreground is an offline missing-tissue diagnostic, never a shifted ROI.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi


def stain_matrix(second):
    h = np.array([.65, .70, .29]); h /= np.linalg.norm(h)
    s = np.asarray(second, dtype=float); s /= np.linalg.norm(s)
    r = np.cross(h, s); r /= np.linalg.norm(r)
    return np.column_stack([h, s, r])


def components(rgb, he=False):
    second = [.07, .99, .11] if he else [.27, .57, .78]
    od = -np.log((np.asarray(rgb, dtype=float) + 1) / 256)
    return np.maximum(od @ np.linalg.inv(stain_matrix(second)).T, 0)


def foreground(rgb, threshold=.20):
    od = -np.log((np.asarray(rgb, dtype=float) + 1) / 256)
    mask = ndi.binary_closing(od.mean(2) > threshold, iterations=2)
    labels, _ = ndi.label(mask)
    sizes = np.bincount(labels.ravel()); keep = sizes >= 32; keep[0] = False
    return keep[labels]


def descriptor(rgb):
    h = components(rgb, he=True)[:, :, 0]
    gradient = np.hypot(*np.gradient(ndi.gaussian_filter(h, 1)))
    peaks = (h == ndi.maximum_filter(h, size=9)) & (h > .25)
    return np.array([h.mean(), h.std(), *np.quantile(h, [.25, .75]),
                     gradient.mean(), gradient.std(), (rgb.min(2) > 205).mean(),
                     peaks.mean(), foreground(rgb).mean()])


def to_tensor(rgb):
    return torch.from_numpy(np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1)


def to_rgb(tensor):
    return ((tensor.detach().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)


class StainStatistics(nn.Module):
    def __init__(self, window=128, threshold=.20, sharpness=25.):
        super().__init__()
        if window <= 0 or window % 4:
            raise ValueError('Statistic window must be positive and divisible by four.')
        self.window, self.threshold, self.sharpness = window, threshold, sharpness
        inverse = np.linalg.inv(stain_matrix([.27, .57, .78]))
        self.register_buffer('dab_vector', torch.tensor(inverse[1], dtype=torch.float32))

    def dab(self, image):
        rgb = (image.clamp(-1, 1) + 1) * 127.5
        od = -torch.log((rgb + 1) / 256)
        return torch.einsum('c,bchw->bhw', self.dab_vector, od).clamp_min(0).unsqueeze(1)

    def forward(self, image, mask, soft=False):
        if image.shape[-2:] != mask.shape[-2:] or mask.shape[1] != 1:
            raise ValueError('Image and fixed mask must have identical spatial dimensions.')
        if any(s % self.window for s in image.shape[-2:]):
            raise ValueError('Input dimensions must be multiples of the statistic window.')
        d = self.dab(image)
        positive = torch.sigmoid((d - self.threshold) * self.sharpness) if soft else (d > self.threshold).float()
        denominator = F.avg_pool2d(mask.float(), self.window)
        numerator = F.avg_pool2d(positive * mask, self.window)
        valid = denominator > 0
        return numerator / denominator.clamp_min(1e-8), valid

    @staticmethod
    def bins(q, count=5):
        return (q.clamp(0, 1) * count).long().clamp_max(count - 1).squeeze(1)


def masked_mean(value, valid):
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1)
