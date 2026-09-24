"""Training/development-only frozen visual representation diagnostic."""
import json
import numpy as np
import torch
from safetensors.torch import load_file
from mmso.artifacts import ROOT,read_manifest,sha256,write_json
from mmso.joint_data import MANIFEST
from mmso.joint_model import NativeDecisionModel,image_tensor
from mmso.joint_world import WORDS

torch.set_num_threads(1)
checkpoint=ROOT/'artifacts/joint-full-v1/model.safetensors'
config=json.loads(checkpoint.with_name('config.json').read_text())
model=NativeDecisionModel(len(config['vocabulary']),config['width'],config['layers'])
model.load_state_dict(load_file(str(checkpoint)));model.eval()
scenes=read_manifest(MANIFEST)


def extract(split,limit):
    selected=[s for s in scenes if s['split']==split][:limit];features=[];targets=[]
    for start in range(0,len(selected),32):
        batch=selected[start:start+32]
        image=torch.stack([image_tensor(ROOT/s['image']['path']) for s in batch])
        patches=torch.stack([image[:,:,:64,:64],image[:,:,:64,64:],image[:,:,64:,:64],image[:,:,64:,64:]],1)
        with torch.inference_mode():features.append(model.image(patches.reshape(-1,3,64,64)).numpy())
        targets.extend(WORDS.index(t['word']) for s in batch for t in s['panel']['tiles'])
    return np.concatenate(features).astype('float64'),np.array(targets)


x,y=extract('train',512);v,vy=extract('dev',192)
mean=x.mean(0);std=np.maximum(x.std(0),1e-4)
x=np.column_stack([(x-mean)/std,np.ones(len(x))]);v=np.column_stack([(v-mean)/std,np.ones(len(v))])
weights=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@np.eye(8)[y])
result={'kind':'frozen_visual_representation_diagnostic','source_run':'joint-full-v1',
        'source_checkpoint_sha256':sha256(checkpoint),'train_patches':len(x),'dev_patches':len(v),
        'train_accuracy':float(((x@weights).argmax(1)==y).mean()),
        'development_accuracy':float(((v@weights).argmax(1)==vy).mean()),
        'method':'Ridge linear probe, lambda=1, normalization fitted on training patches only',
        'test_used':False,'checkpoint_saved':False}
write_json(ROOT/'reports/joint-full-v1/visual-probe.json',result)
print(json.dumps(result))
