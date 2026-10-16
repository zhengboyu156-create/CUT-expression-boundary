"""CUT + regional DAB-distribution head + FiLM + explicit proxy supervision."""
import torch
from torch.nn import functional as F
from .cut_model import CUTModel
from expression.networks import ConditionalGenerator
from expression.statistics import StainStatistics, masked_mean
from util.util import str2bool


class ExpressionCUTModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser,is_train=True,command_line=None):
        parser=CUTModel.modify_commandline_options(parser,is_train,command_line)
        parser.add_argument('--stats_window',type=int,default=128)
        parser.add_argument('--lambda_distribution',type=float,default=1.)
        parser.add_argument('--lambda_statistics',type=float,default=10.)
        parser.add_argument('--lambda_moment',type=float,default=0.,help='Experimental; disabled after multi-pair regression.')
        parser.add_argument('--teacher_probability',type=float,default=1.,help='Experimental prediction mixing when below 1.')
        parser.add_argument('--learn_bin_means',type=str2bool,default=False,help='Experimental within-bin representatives.')
        parser.add_argument('--lambda_pixel_proxy',type=float,default=.1,
                            help='Pixel DAB pseudo-label loss, not cell-positive supervision.')
        parser.add_argument('--lambda_paired_l1',type=float,default=0.)
        parser.add_argument('--backbone_checkpoint',default='')
        parser.set_defaults(netG='resnet_6blocks',dataset_mode='expression_pair',preprocess='none')
        return parser

    def __init__(self,opt):
        if opt.netG!='resnet_6blocks' or opt.flip_equivariance:
            raise ValueError('This implementation requires resnet_6blocks and loader-synchronized flips.')
        if not 0<=getattr(opt,'teacher_probability',.5)<=1:raise ValueError('Teacher probability must lie in [0,1].')
        super().__init__(opt)
        if opt.backbone_checkpoint:
            self.netG.load_state_dict(torch.load(opt.backbone_checkpoint,map_location=self.device,weights_only=True))
        self.netG=ConditionalGenerator(self.netG,opt.ngf*4,opt.stats_window,learn_bin_means=getattr(opt,'learn_bin_means',False)).to(self.device)
        self.training_progress=0.
        self.statistic=StainStatistics(opt.stats_window).to(self.device)
        self.loss_names+=['distribution','moment','statistics','pixel_proxy','paired_l1']
        if self.isTrain:
            # The original optimizer predates the new heads; rebuild it explicitly.
            old_optimizer=self.optimizer_G
            self.optimizer_G=torch.optim.Adam(self.netG.parameters(),lr=opt.lr,betas=(opt.beta1,opt.beta2))
            self.optimizers=[self.optimizer_G if o is old_optimizer else o for o in self.optimizers]

    def set_input(self,data):
        super().set_input(data)
        if data['bbox'].shape!=data['paired_bbox'].shape or not torch.equal(data['bbox'],data['paired_bbox']):
            raise ValueError('Paired reference coordinates changed.')
        self.paired_B=data['paired_B'].to(self.device)
        self.fixed_mask=data['tissue_mask'].to(self.device)
        with torch.no_grad():
            self.target_q,self.valid=self.statistic(self.paired_B,self.fixed_mask)

    def data_dependent_initialize(self,data):
        # Keep all synchronized tensors together; the original method slices only A/B.
        self.set_input(data);self.forward()
        if self.isTrain:
            self.compute_D_loss().backward();self.compute_G_loss().backward()
            if self.opt.lambda_NCE>0:
                self.optimizer_F=torch.optim.Adam(self.netF.parameters(),lr=self.opt.lr,betas=(self.opt.beta1,self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def forward(self):
        teacher=self.target_q if self.isTrain else None
        teacher_mask=None
        if self.isTrain:
            # First 20%: establish the statistic-to-color mapping. Then gradually
            # expose G to its H&E-only predicted conditions, as used at deployment.
            progress=min(1.,max(0.,(self.training_progress-.2)/.6))
            probability=1-progress*(1-getattr(self.opt,'teacher_probability',1.))
            teacher_mask=torch.ones_like(teacher,dtype=torch.bool) if probability>=1 else torch.rand_like(teacher)<probability
        self.fake_B,self.expression_aux=self.netG(self.real_A,condition=teacher,teacher_mask=teacher_mask,return_aux=True)
        if self.opt.nce_idt and self.isTrain:
            # Identity branch remains separate from the H&E posterior supervision.
            with torch.no_grad():
                idt_mask=torch.ones_like(self.real_B[:,:1])
                idt_q,_=self.statistic(self.real_B,idt_mask)
            self.idt_B=self.netG(self.real_B,condition=idt_q)

    def compute_G_loss(self):
        original=super().compute_G_loss()
        cross_entropy=F.cross_entropy(self.expression_aux['logits'],self.statistic.bins(self.target_q),reduction='none')
        self.loss_distribution=masked_mean(cross_entropy,self.valid.squeeze(1))*self.opt.lambda_distribution
        self.loss_moment=masked_mean((self.expression_aux['expected_condition']-self.target_q).abs(),self.valid)*getattr(self.opt,'lambda_moment',0.)
        generated_q,_=self.statistic(self.fake_B,self.fixed_mask,soft=True)
        self.loss_statistics=masked_mean((generated_q-self.expression_aux['condition']).abs(),self.valid)*self.opt.lambda_statistics
        pixel_target=(self.statistic.dab(self.paired_B)>.20).float()
        pixel_loss=F.binary_cross_entropy_with_logits(self.expression_aux['pixel_logits'],pixel_target,reduction='none')
        self.loss_pixel_proxy=masked_mean(pixel_loss,self.fixed_mask)*self.opt.lambda_pixel_proxy
        self.loss_paired_l1=masked_mean((self.fake_B-self.paired_B).abs().mean(1,keepdim=True),self.fixed_mask)*self.opt.lambda_paired_l1
        self.loss_G=original+self.loss_distribution+self.loss_moment+self.loss_statistics+self.loss_pixel_proxy+self.loss_paired_l1
        return self.loss_G

    def set_training_progress(self,fraction):
        self.training_progress=min(1.,max(0.,float(fraction)))
