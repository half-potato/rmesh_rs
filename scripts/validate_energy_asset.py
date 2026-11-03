#!/usr/bin/env python3
"""Independent energy-v1 container checks and checkpoint material-field probes."""
import argparse,gzip,json,struct,sys,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('asset',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--sdir',type=Path);p.add_argument('--mesh-source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
raw=gzip.decompress(a.asset.read_bytes());nv,nt,degree,k=struct.unpack_from('<4I',raw);assert degree==0
pos=48;vertices=np.frombuffer(raw,'<f4',nv*3,pos).reshape(nv,3);pos+=nv*12
indices=np.frombuffer(raw,'<i4',nt*4,pos).reshape(nt,4);pos+=nt*16
assert indices.min()>=0 and indices.max()<nv and np.isfinite(vertices).all()
pos=(pos+nt+3)&~3;pos=(pos+(3+min(k,3)*3+nt*min(k,3))*2+3)&~3;pos=(pos+nt*3*2+3)&~3
magic,count=struct.unpack_from('<2I',raw,pos);pos+=8;assert magic==0x524d5458
sections={}
for _ in range(count):
    tag=raw[pos:pos+16].rstrip(b'\0').decode();pos+=16;dtype,rank=struct.unpack_from('<2I',raw,pos);pos+=8
    shape=struct.unpack_from('<'+'I'*rank,raw,pos);pos+=rank*4;length,=struct.unpack_from('<I',raw,pos);pos+=4
    kind=['<f2','<f4','u1','<i4'][dtype];assert length==int(np.prod(shape))*np.dtype(kind).itemsize
    assert tag not in sections;sections[tag]=np.frombuffer(raw,kind,int(np.prod(shape)),pos).reshape(shape);pos=(pos+length+3)&~3
assert pos==len(raw);assert sections['energy_version'].tolist()==[1,4,1]
expected={'density_f32':(nt,),'albedo':(nt,3),'roughness':(nt,),'fresnel_rgb':(nt,3),'backscatter':(nt,),'geom_normals':(nv,3),'vertex_normals':(nv,3),'ggx_table':(2,65,65),'ggx_average':(2,65)}
for key,shape in expected.items():assert sections[key].shape==shape and sections[key].dtype==np.float32 and np.isfinite(sections[key]).all(),key
for key in ['albedo','fresnel_rgb','backscatter']:assert sections[key].min()>=0 and sections[key].max()<=1,key
assert sections['roughness'].min()>=.04 and sections['roughness'].max()<=1 and sections['density_f32'].min()>=0
lut=np.concatenate([sections['ggx_table'],np.broadcast_to(sections['ggx_average'][:,:,None],(2,65,65))]).transpose(1,2,0).astype('<f4')
report={'vertices':nv,'tets':nt,'sections':count,'version':[1,4,1],'density_sum':float(sections['density_f32'].sum(dtype=np.float64)),'embedded_reference_lut_exact':lut.tobytes()==Path('crates/rmesh-viewer-core/src/assets/ggx_energy.rgba32f').read_bytes()}
if a.checkpoint:
    import torch,types
    from sdir_checkpoint_reference import hash_encoder,load_module
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    root=a.mesh_source.resolve();sys.path[:0]=[str(root),str(a.sdir)]
    package=types.ModuleType('radiance_meshes');package.__path__=[str(root)];sys.modules['radiance_meshes']=package
    from utils.hashgrid import HashEmbedderOptimized
    cfg=json.loads((a.checkpoint/'config.json').read_text());assert cfg['material_model']=='energy'
    saved=torch.load(a.checkpoint/'flash_material.pt',map_location='cpu',weights_only=True);assert saved['mode']==4
    module=load_module('sdir_energy_field_probe',a.sdir/'renderer/flash_material_field.py')
    field=module.FlashMaterialField.__new__(module.FlashMaterialField);torch.nn.Module.__init__(field)
    for key in ['lower','upper','initial']:field.register_buffer(key,saved['field'][key])
    field.encoding=hash_encoder(HashEmbedderOptimized,saved['field'],'encoding.',18)
    ids=np.random.default_rng(43).choice(nt,32,replace=False)
    centers=torch.from_numpy(vertices[indices[ids]].copy()).mean(1)
    values=field(centers)[1].numpy();target=np.column_stack([sections['roughness'][ids],sections['fresnel_rgb'][ids],sections['backscatter'][ids]])
    error=float(abs(values-target).max());report['checkpoint_material_probe_count']=len(ids);report['checkpoint_material_max_abs_error']=error
    assert error<=.01,report
report['passed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
