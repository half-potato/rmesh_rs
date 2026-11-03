#!/usr/bin/env python3
"""Freeze direct-light shader cases from the actual sdir PyTorch implementation."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--sdir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.sdir));from renderer.energy_material import GGXEnergy,ggx_unit
torch.manual_seed(43);torch.set_grad_enabled(False)
count=2048
n=F.normalize(torch.randn(count,3),dim=-1);l=F.normalize(torch.randn(count,3),dim=-1);v=F.normalize(torch.randn(count,3),dim=-1)
r=.04+.96*torch.rand(count);f0=torch.rand(count,3);albedo=torch.rand(count,3);b=torch.rand(count,1)
# Exact backfaces, zero/full F0, white/black albedo, roughness bounds,
# smooth aligned highlights and grazing angles.
n[:64]=torch.tensor([0.,0.,1.]);l[:64]=n[:64];v[:64]=n[:64]
r[:64]=torch.linspace(.04,1.,64);f0[:16]=0;f0[16:32]=1;b[:48]=0;b[48:64]=1
l[64:96]=-n[64:96];v[96:128]=-n[96:128]
albedo[128:144]=0;albedo[144:160]=1
for i in range(160,192):
    n[i]=torch.tensor([0.,0.,1.]);l[i]=F.normalize(torch.tensor([1.,0.,1e-5*(i-159)]),dim=0);v[i]=l[i]*torch.tensor([-1.,1.,1.]);r[i]=.4
energy=GGXEnergy();d,s=energy.brdf(n,l,v,r,f0,albedo)
back=ggx_unit(n,l,2*(n*v).sum(-1,keepdim=True)*n-v,torch.full_like(r,.25))[:,None]
expected=((1-b)*(d+s)+b*back)*(n*l).sum(-1).clamp_min(0)[:,None]*torch.pi
packed=torch.zeros(count,6,4);packed[:,0,:3]=n;packed[:,0,3]=r;packed[:,1,:3]=l;packed[:,1,3]=b[:,0];packed[:,2,:3]=v;packed[:,3,:3]=f0;packed[:,4,:3]=albedo;packed[:,5,:3]=expected
packed.numpy().astype('<f4').tofile(a.output)
meta={'source':str(a.sdir/'renderer/energy_material.py'),'sha256':hashlib.sha256((a.sdir/'renderer/energy_material.py').read_bytes()).hexdigest(),'cases':count,'seed':43,'layout':'6 vec4: normal+roughness, light+retro, view, F0, albedo, expected source','max_source':float(expected.max())}
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
