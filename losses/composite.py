from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _focal_frequency_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """A lightweight FFL-style spectrum regularizer.

    It compares the log magnitude spectra of the prediction and target in the
    frequency domain and emphasizes large spectral disagreement with a focal
    weighting shape. This is intentionally not a full paper implementation, but
    it provides a differentiable, config-driven frequency signal that can help
    guard against reconstruction that is too smooth in the Fourier domain.
    """
    eps = 1e-6
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))
    pred_mag = torch.abs(pred_fft).clamp_min(eps)
    target_mag = torch.abs(target_fft).clamp_min(eps)

    # log-magnitude spectra
    pred_log_mag = torch.log(pred_mag)
    target_log_mag = torch.log(target_mag)
    log_mag_diff = (pred_log_mag - target_log_mag).abs()

    # Focal weighting: low values are down-weighted, larger spectral errors
    # receive a stronger penalty. This maps the algebraic intent of FFL onto a
    # stable differentiable tensor expression.
    focal_weights = log_mag_diff.pow(beta)
    return (focal_weights * log_mag_diff).mean()

def _gaussian(ch, device, dtype, size=11, sigma=1.5):
    x=torch.arange(size,device=device,dtype=dtype)-size//2; g=torch.exp(-x.square()/(2*sigma*sigma)); g=(g/g.sum())[:,None]@(g/g.sum())[None,:]
    return g.expand(ch,1,size,size)
def ssim(x,y):
    c1,c2=.01**2,.03**2; k=_gaussian(x.shape[1],x.device,x.dtype); mu1=F.conv2d(x,k,padding=5,groups=x.shape[1]); mu2=F.conv2d(y,k,padding=5,groups=y.shape[1])
    a=F.conv2d(x*x,k,padding=5,groups=x.shape[1])-mu1*mu1; b=F.conv2d(y*y,k,padding=5,groups=y.shape[1])-mu2*mu2; ab=F.conv2d(x*y,k,padding=5,groups=x.shape[1])-mu1*mu2
    return (((2*mu1*mu2+c1)*(2*ab+c2))/((mu1.square()+mu2.square()+c1)*(a+b+c2))).mean((1,2,3))
def ms_ssim(x,y, levels=4):
    vals=[]
    for _ in range(levels):
        vals.append(ssim(x,y));
        if min(x.shape[-2:]) < 4: break
        x,y=F.avg_pool2d(x,2),F.avg_pool2d(y,2)
    return torch.stack(vals).prod(0)
def psnr(x,y): return -10*torch.log10(F.mse_loss(x.clamp(0,1),y.clamp(0,1))+1e-10)
class CompositeLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.sobel_x=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32)[None,None]
        self.perceptual = None
        if cfg.get("perceptual_enabled", False):
            from torchvision.models import VGG16_Weights, vgg16
            net = vgg16(weights=VGG16_Weights.DEFAULT).features[:9].eval()
            for p in net.parameters(): p.requires_grad = False
            self.perceptual = net
    def forward(self,pred,target,warmup=False):
        charb=torch.sqrt((pred-target).square()+1e-6).mean()
        if warmup:return charb,{"charbonnier":charb.detach()}
        k=self.sobel_x.to(pred); grad=(F.conv2d(pred,k,padding=1)-F.conv2d(target,k,padding=1)).abs().mean()+(F.conv2d(pred,k.transpose(-1,-2),padding=1)-F.conv2d(target,k.transpose(-1,-2),padding=1)).abs().mean()
        mss=1-ms_ssim(pred.clamp(0,1),target).mean()
        focal = _focal_frequency_loss(pred, target, beta=float(self.cfg.get("focal_beta", 2.0)))
        total=self.cfg.get("charbonnier",1)*charb+self.cfg.get("ms_ssim",.3)*mss+self.cfg.get("gradient",.1)*grad+self.cfg.get("focal_frequency",.1)*focal
        perceptual = pred.new_zeros(())
        if self.perceptual is not None:
            # VGG expects RGB; retain gradients only through the prediction branch.
            mean=torch.tensor([.485,.456,.406],device=pred.device)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=pred.device)[None,:,None,None]
            a=(pred.clamp(0,1).repeat(1,3,1,1)-mean)/std; b=(target.repeat(1,3,1,1)-mean)/std
            perceptual=F.l1_loss(self.perceptual(a),self.perceptual(b).detach()); total=total+self.cfg.get("perceptual",.05)*perceptual
        return total,{"charbonnier":charb.detach(),"ms_ssim":mss.detach(),"gradient":grad.detach(),"focal_frequency":focal.detach(),"perceptual":perceptual.detach()}
