from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from models import RestorerConfig, WindowRestorer

def transforms(x):
    yield x, lambda y:y
    yield x.flip(-1), lambda y:y.flip(-1)
    yield x.flip(-2), lambda y:y.flip(-2)
    yield x.transpose(-1,-2), lambda y:y.transpose(-1,-2)
def tiled(model,x,tile,overlap):
    _,_,h,w=x.shape; tile=min(tile,h,w); overlap=min(overlap,tile-1); s=model.cfg.scale; out=torch.zeros(1,1,h*s,w*s,device=x.device); weight=torch.zeros_like(out); stride=tile-overlap
    for y in range(0,h,stride):
      for z in range(0,w,stride):
        yy=min(y,h-tile); zz=min(z,w-tile); p=model(x[:,:,yy:yy+tile,zz:zz+tile]); out[:,:,yy*s:(yy+tile)*s,zz*s:(zz+tile)*s]+=p; weight[:,:,yy*s:(yy+tile)*s,zz*s:(zz+tile)*s]+=1
    return out/weight.clamp_min(1)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--tta',action='store_true'); ap.add_argument('--tile',type=int,default=0); ap.add_argument('--overlap',type=int,default=16); a=ap.parse_args()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model=WindowRestorer(RestorerConfig(**ck['model_config'])).cuda() if torch.cuda.is_available() else WindowRestorer(RestorerConfig(**ck['model_config']))
 model.load_state_dict(ck['model']); model.eval(); dev=next(model.parameters()).device; Path(a.output).mkdir(parents=True,exist_ok=True)
 for p in sorted(Path(a.input).glob('*.npy')):
  x=torch.from_numpy(np.load(p).astype('float32'))[None,None].to(dev); t=time.perf_counter()
  with torch.no_grad():
   if a.tta: pred=sum(inv(tiled(model,z,a.tile,a.overlap) if a.tile else model(z)) for z,inv in transforms(x))/4
   else: pred=tiled(model,x,a.tile,a.overlap) if a.tile else model(x)
  elapsed=time.perf_counter()-t; np.save(Path(a.output)/p.name,pred[0,0].clamp(0,1).cpu().numpy()); print(f'{p.name}: {elapsed:.3f}s' + (' WARNING: slow' if elapsed>5 else ''))
if __name__=='__main__': main()
