from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class RestorerConfig:
    scale: int = 2; embed_dim: int = 192; depth: int = 12; heads: int = 6; window_size: int = 8; mlp_ratio: float = 4.; noise_conditioning: bool = True

def _windows(x, ws):
    b,h,w,c=x.shape; return x.view(b,h//ws,ws,w//ws,ws,c).permute(0,1,3,2,4,5).reshape(-1,ws*ws,c)
def _unwindows(x, b,h,w,ws):
    return x.view(b,h//ws,w//ws,ws,ws,-1).permute(0,1,3,2,4,5).reshape(b,h,w,-1)

class SwinBlock(nn.Module):
    def __init__(self, d, heads, ws, shift, ratio):
        super().__init__(); self.ws,self.shift=ws,shift; self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True); self.n2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,int(d*ratio)),nn.GELU(),nn.Linear(int(d*ratio),d)); self.local=nn.Conv2d(d,d,3,1,1,groups=d)
    def forward(self,x):
        b,c,h,w=x.shape; pad_h=(self.ws-h%self.ws)%self.ws; pad_w=(self.ws-w%self.ws)%self.ws
        z=F.pad(x,(0,pad_w,0,pad_h)); hp,wp=z.shape[-2:]; z=z.permute(0,2,3,1)
        if self.shift: z=torch.roll(z,(-self.shift,-self.shift),(1,2))
        q=_windows(z,self.ws); a=self.attn(self.n1(q),self.n1(q),self.n1(q),need_weights=False)[0]; z=_unwindows(q+a,b,hp,wp,self.ws)
        if self.shift: z=torch.roll(z,(self.shift,self.shift),(1,2))
        z=z.permute(0,3,1,2); z=z+self.local(z); z=z.permute(0,2,3,1); z=z+self.mlp(self.n2(z)); return z[:,:h,:w].permute(0,3,1,2)

class WindowRestorer(nn.Module):
    def __init__(self, cfg: RestorerConfig):
        super().__init__(); self.cfg=cfg; d=cfg.embed_dim
        self.noise = nn.Sequential(nn.Conv2d(1,d//2,3,1,1),nn.GELU(),nn.Conv2d(d//2,1,3,1,1)) if cfg.noise_conditioning else None
        self.stem=nn.Sequential(nn.Conv2d(2 if self.noise else 1,d,3,1,1),nn.GELU(),nn.Conv2d(d,d,3,1,1))
        self.blocks=nn.Sequential(*[SwinBlock(d,cfg.heads,cfg.window_size,0 if i%2==0 else cfg.window_size//2,cfg.mlp_ratio) for i in range(cfg.depth)])
        self.body=nn.Conv2d(d,d,3,1,1); layers=[]; steps=1 if cfg.scale==2 else 2
        for _ in range(steps): layers += [nn.Conv2d(d,d*4,3,1,1),nn.PixelShuffle(2),nn.GELU()]
        self.up=nn.Sequential(*layers); self.out=nn.Conv2d(d,1,3,1,1)
    def forward(self,x, return_intermediate=False):
        base=F.interpolate(x,scale_factor=self.cfg.scale,mode='bicubic',align_corners=False); inp=torch.cat([x,self.noise(x)],1) if self.noise else x
        stem=self.stem(inp); f=self.body(self.blocks(stem))+stem; y=self.out(self.up(f))+base
        return y
