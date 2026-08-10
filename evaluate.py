from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np, torch
from losses import psnr, ssim, ms_ssim
def main():
 a=argparse.ArgumentParser(); a.add_argument('--predictions',required=True); a.add_argument('--gt',required=True); a.add_argument('--report',default='report.csv'); z=a.parse_args(); rows=[]
 for p in sorted(Path(z.predictions).glob('*.npy')):
  q=Path(z.gt)/p.name
  if not q.exists(): continue
  pred=torch.from_numpy(np.load(p).astype('float32'))[None,None]; gt=torch.from_numpy(np.load(q).astype('float32'))[None,None]
  edge=(gt[:,:,:,1:]-gt[:,:,:,:-1]).abs().mean().item(); rows.append([p.name,psnr(pred,gt).item(),ssim(pred,gt).item(),ms_ssim(pred,gt).item(),edge])
 with open(z.report,'w',newline='') as f:
  w=csv.writer(f); w.writerow(['image','psnr','ssim','ms_ssim','edge_density']); w.writerows(rows)
 r=np.asarray([x[1:] for x in rows],float); print({'n':len(rows),'psnr':r[:,0].mean(),'ssim':r[:,1].mean(),'ms_ssim':r[:,2].mean(),'complex_ssim':r[r[:,3]>=np.median(r[:,3]),1].mean()})
if __name__=='__main__': main()
