#!/usr/bin/env python3
"""Run sdir's actual Warp quadrature/shading on CPU against a supplied checkpoint.

Requires torch, warp-lang, numpy, scipy, icecream and plyfile. --mesh-source is
an unpacked radiance_meshes source tree at sdir's submodule revision. Nothing in
sdir/checkpoint is modified. Only the CPU-compatible reference functions are
loaded; CUDA training/model initialization is deliberately not invoked.
"""
import argparse
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from plyfile import PlyData


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def hash_encoder(cls, state, prefix, log_size):
    """Use the reference eager encoder with the checkpoint's own tables."""
    enc = cls.__new__(cls)
    torch.nn.Module.__init__(enc)
    for key in ['box_min','box_max','base_resolution','finest_resolution','level_resolutions','primes','box_offsets']:
        enc.register_buffer(key, state[prefix+key])
    enc.n_levels = len(enc.level_resolutions)
    enc.use_conditional_hashing = True
    enc.log2_hashmap_size = log_size
    enc.hash_mask = (1 << log_size) - 1
    enc.embeddings = torch.nn.ModuleList([
        torch.nn.Embedding.from_pretrained(state[f'{prefix}embeddings.{i}.weight'])
        for i in range(enc.n_levels)])
    return enc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sdir', type=Path, required=True)
    p.add_argument('--mesh-source', type=Path, required=True)
    p.add_argument('--camera-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--export-scene', type=Path, help='Export lossless full-precision evaluated checkpoint for the native reference path')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--height', type=int, default=18)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--mode', type=int, choices=[0,3,4], default=3)
    args = p.parse_args()
    wp.config.kernel_cache_dir = str(args.output.parent / "warp-cache")
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    root = args.mesh_source.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.sdir.resolve()))
    # Namespace-only packages avoid importing the CUDA extension's __init__.
    for name, relative in [('radiance_meshes',''), ('radiance_meshes.delaunay_rasterization','delaunay_rasterization'),
                           ('radiance_meshes.delaunay_rasterization.internal','delaunay_rasterization/internal')]:
        mod = types.ModuleType(name); mod.__path__ = [str(root/relative)]; sys.modules[name] = mod
    wt = load_module('radiance_meshes.delaunay_rasterization.internal.warp_tracer', root/'delaunay_rasterization/internal/warp_tracer.py')
    vr = load_module('sdir_volumetric_reference', args.sdir/'renderer/volumetric_relight.py')
    from utils.hashgrid import HashEmbedderOptimized
    from utils.contraction import contract_mean_std
    from utils.topo_utils import calculate_circumcenters_torch
    from utils.safe_math import safe_div, safe_sqrt
    print('Loading checkpoint', flush=True)
    state = torch.load(args.checkpoint/'ckpt.pth', map_location='cpu', weights_only=True)
    ref = torch.load(args.checkpoint/'ref_renderer.pt', map_location='cpu', weights_only=True)
    cfg = json.loads((args.checkpoint/'config.json').read_text())
    verts = torch.cat([state['interior_vertices'], state['ext_vertices']]).float().contiguous()
    tets = state['indices'].int().contiguous()
    has_ply = (args.checkpoint/'ckpt.ply').exists()
    if has_ply:
        ply = PlyData.read(args.checkpoint/'ckpt.ply', known_list_len={'tetrahedron': {'indices': 4}})
        assert np.array_equal(ply['tetrahedron']['indices'], tets.numpy()), 'PLY/checkpoint topology differs'
        rho = torch.from_numpy(np.array(ply['tetrahedron']['s'], copy=True)).float()
    else:
        # New energy checkpoints omit the convenience PLY. Evaluate ALL saved
        # cells before camera quadrature; never substitute quantized densities.
        rho = torch.zeros(len(tets))
    assert torch.isfinite(rho).all() and (rho >= 0).all()
    primary = torch.zeros(len(tets),15); primary[:,0] = rho
    normals_fn = source_function(root/'models/ingp_color.py', 'compute_geometric_vertex_normals', {'torch':torch,'F':F})
    print('Computing reference density normals', flush=True)
    normals = normals_fn(types.SimpleNamespace(vertices=verts, indices=tets.long()), density=rho).contiguous() if has_ply else torch.zeros_like(verts)
    print('Building reference face BVH', flush=True)
    tracer = wt.WarpFaceTracer(verts,tets)
    camera = json.loads(args.camera_json.read_text())
    w,h=args.width,args.height
    fx,fy,cx,cy=camera['intrinsics']
    fx*=w/camera['width']; cx*=w/camera['width']; fy*=h/camera['height']; cy*=h/camera['height']
    yy,xx=torch.meshgrid(torch.arange(h)+.5,torch.arange(w)+.5,indexing='ij')
    direction=torch.stack([(xx-cx)/fx,(yy-cy)/fy,torch.ones_like(xx)],-1).reshape(-1,3)
    c2w=torch.tensor(camera['c2w_columns'])[:,:3].T
    direction=F.normalize(direction@c2w.T,dim=-1)
    eye=torch.tensor(camera['camera_position'][:3],dtype=torch.float32)
    rays=torch.cat([eye.expand(len(direction),3),direction],-1).contiguous()
    print('Preparing optical-depth quadrature', flush=True)
    q=vr.prepare_volume(tracer,rays,primary,normals,min_t=camera['near'],samples=args.samples) if has_ply else {'ids':torch.empty(0,dtype=torch.int32)}
    ids=q['ids'].unique().long(); ids=ids[ids>=0]
    if args.export_scene is not None or not has_ply: ids=torch.arange(len(tets))
    print('Evaluating materials for',len(ids),'sampled cells;',len(q['ids']),'samples',flush=True)
    enc=hash_encoder(HashEmbedderOptimized,state,'backbone._orig_mod.encoding.',cfg['log2_hashmap_size'])
    def head(x,name):
        for i in [0,2,4]:
            prefix=f'backbone._orig_mod.{name}.{i}.'
            x=F.linear(x,state[prefix+'weight'],state[prefix+'bias'])
            if i<4: x=F.selu(x)
        return x
    raw=torch.zeros(len(ids),15)
    density_error=[]
    for begin in range(0,len(ids),16384):
        subset=ids[begin:begin+16384]
        cc,radius=calculate_circumcenters_torch(verts[tets[subset].long()].double())
        cv,cr=contract_mean_std((cc-state['center'])/state['scene_scaling'],radius/state['scene_scaling'])
        x=((cv/2+1)/2).float()
        embedding=enc(x).reshape(-1,cfg['hashmap_dim'],cfg['L'])
        n=torch.arange(cfg['L']).reshape(1,1,-1)
        erf_x=safe_div(torch.tensor(1.),safe_sqrt(cfg['per_level_scale']*4*n*(cr*cfg['scale_multi']).reshape(-1,1,1)))
        scaling=torch.sign(erf_x)*torch.sqrt(1-torch.exp(-(4/torch.pi)*erf_x**2))
        features=(embedding*scaling).reshape(-1,cfg['L']*cfg['hashmap_dim']).float()
        raw[begin:begin+len(subset),7:]=head(features,'network')
        # Model consumes density_offset itself; its backbone receives the iNGPDW default -4.
        predicted=torch.exp(head(features,'density_net').squeeze(-1)-4.0)
        density_error.append(((predicted-rho[subset]).abs()/rho[subset].clamp_min(1e-10)).numpy())
        if args.export_scene is not None or not has_ply: rho[subset]=predicted
        if begin % 262144 == 0: print('Material cells',begin,'/',len(ids),flush=True)
    raw[:,0]=rho[ids]
    activate=source_function(args.sdir/'renderer/ref_renderer.py','activate_aux',{'torch':torch})
    primary[ids]=activate(raw,types.SimpleNamespace(rough_global=ref['rough_global'],albedo_global=ref['albedo_global']))
    relative=np.concatenate(density_error)
    if has_ply: print('PLY vs CPU checkpoint density relative error p50/p99/max:',np.quantile(relative,[.5,.99,1]),flush=True)
    material=None
    if args.mode in (3,4):
        saved=torch.load(args.checkpoint/'flash_material.pt',map_location='cpu',weights_only=True)
        assert saved['mode']==args.mode, 'Checkpoint material semantics do not match requested mode'
        field_mod=load_module('sdir_flash_material',args.sdir/'renderer/flash_material_field.py')
        field=field_mod.FlashMaterialField.__new__(field_mod.FlashMaterialField);torch.nn.Module.__init__(field)
        for k in ['lower','upper','initial']: field.register_buffer(k,saved['field'][k])
        field.encoding=hash_encoder(HashEmbedderOptimized,saved['field'],'encoding.',18)
        material=torch.zeros(len(tets),5)
        residual_max=0.
        for subset in ids.split(16384):
            residual,values=field(verts[tets[subset].long()].mean(1))
            residual_max=max(residual_max,float(residual.abs().max()))
            material[subset]=values
        print('M3 albedo residual max (not applied by canonical volumetric_relight):',residual_max,flush=True)
    if args.export_scene is not None or not has_ply:
        primary[:,0]=rho
        normals=normals_fn(types.SimpleNamespace(vertices=verts,indices=tets.long()),density=rho).contiguous()
        q=vr.prepare_volume(tracer,rays,primary,normals,min_t=camera['near'],samples=args.samples)
    if args.export_scene is not None:
        table=torch.cat([primary[:,7:8],primary[:,12:15],material if material is not None else torch.zeros(len(tets),5)],1)
        import struct
        extent=float((verts.max(0).values-verts.min(0).values).norm())
        with open(args.export_scene,'wb') as f:
            f.write(b'RMLIT001');f.write(struct.pack('<IIIf',len(verts),len(tets),args.mode,max(extent*1e-7,1e-7)))
            for array in [verts,normals,tets,rho,table]: array.numpy().astype('<i4' if array.dtype==torch.int32 else '<f4').tofile(f)
        print('Exported lossless reference scene',args.export_scene,flush=True)
    rays.numpy().astype('<f4').tofile(str(args.output)+'.rays.bin')
    result=torch.zeros(h*w,3)
    for light_index,(position,axis,rgb) in enumerate(camera['light_position_direction_rgb']):
        intensity=camera['light_intensities'][light_index]
        if intensity<=0: continue
        lamp=types.SimpleNamespace(type='point',position=torch.tensor(position),direction=torch.tensor(axis),color=torch.tensor(rgb)*intensity,hemi_soft=0.)
        print('Shading reference light',light_index,flush=True)
        result+=vr.shade_volume(tracer,primary,primary,q,lamp,material=material,eye=eye,material_mode=args.mode)[:,:3]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    image=result.reshape(h,w,3).permute(2,0,1).numpy()
    np.save(args.output,image)
    np.savez(str(args.output)+'.samples.npz',points=q['points'].numpy(),normals=q['normals'].numpy(),ids=q['ids'].numpy(),weights=q['weights'].numpy(),counts=q['counts'].numpy(),primary=primary[ids].numpy(),primary_ids=ids.numpy(),material=material[ids].numpy() if material is not None else np.zeros((len(ids),5)))
    print('Wrote',args.output,'range',float(image.min()),float(image.max()),flush=True)


if __name__=='__main__':
    main()
