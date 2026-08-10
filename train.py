"""Two-stage training entry point for the semiconductor restoration model."""
from __future__ import annotations
import argparse, math, random
from pathlib import Path
import numpy as np, torch, yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from datasets import build_loaders
from models import RestorerConfig, WindowRestorer
from losses import CompositeLoss, psnr, ssim, ms_ssim

def validate(model, loader, device):
    model.eval(); rows=[]
    with torch.no_grad():
        for b in loader:
            x,y=b['input'].to(device),b['target'].to(device); p=model(x).clamp(0,1)
            edge=(y[:,:,:,1:]-y[:,:,:,:-1]).abs().mean().item()
            rows.append((psnr(p,y).item(),ssim(p,y).item(),ms_ssim(p,y).item(),edge))
    a=np.asarray(rows); med=np.median(a[:,3]); out={"psnr":a[:,0].mean(),"ssim":a[:,1].mean(),"ms_ssim":a[:,2].mean()}
    # Edge density is a metadata-free proxy for complex/OOD-like structures.
    for tag, mask in [("simple",a[:,3]<med),("complex",a[:,3]>=med)]:
        out[f"{tag}_ssim"]=a[mask,1].mean() if mask.any() else float('nan')
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config))
    seed=cfg.get('seed',42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); mcfg=RestorerConfig(**cfg['model']); model=WindowRestorer(mcfg).to(device)
    train,val=build_loaders(cfg,mcfg.scale); criterion=CompositeLoss(cfg['loss']).to(device); opt=AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    sched=CosineAnnealingLR(opt,T_max=cfg['training']['epochs']); amp=torch.amp.GradScaler('cuda',enabled=cfg['training'].get('amp',True) and device.type=='cuda')
    out=Path(cfg['training']['checkpoint_dir']); out.mkdir(parents=True,exist_ok=True); best=-float('inf')
    global_step=0
    epochs = cfg['training']['epochs']
    for epoch in range(epochs):
        model.train(); warm=epoch<cfg['training']['warmup_epochs']
        if epoch==cfg['training']['warmup_epochs']:
            for group in opt.param_groups: group['lr']=cfg['training']['fine_tune_lr']
        progress = tqdm(
            train,
            desc=f"Epoch {epoch + 1}/{epochs} [{'warmup' if warm else 'composite'}]",
            unit="batch",
            dynamic_ncols=True,
        )
        for step,b in enumerate(progress, start=1):
            if global_step < cfg['training']['warmup_steps']:
                for group in opt.param_groups: group['lr']=cfg['training']['lr']*(global_step+1)/cfg['training']['warmup_steps']
            x,y=b['input'].to(device,non_blocking=True),b['target'].to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp.is_enabled()): p=model(x); loss,_=criterion(p,y,warm)
            amp.scale(loss).backward(); amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']); amp.step(opt); amp.update(); global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        sched.step(); metrics=validate(model,val,device); print(f"epoch={epoch+1} phase={'warmup' if warm else 'composite'} {metrics}")
        state={"model":model.state_dict(),"model_config":mcfg.__dict__,"config":cfg,"epoch":epoch+1,"metrics":metrics}
        torch.save(state,out/'last.pt')
        if metrics['ssim']>best: best=metrics['ssim']; torch.save(state,out/'best_ssim.pt')
if __name__=='__main__': main()
