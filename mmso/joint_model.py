"""Native audio/image/question fusion with one shared candidate scoring head.

No task IDs, scene metadata, transcripts, oracle labels, or candidate IDs enter
the neural network. This first model uses a small learned text vocabulary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import load_file

from .artifacts import ROOT
from .audio import AudioCNN, LogMel, choose_device, read_audio


def words(text):
    return re.findall(r"[a-z]+",text.lower())


class Vocabulary:
    def __init__(self,tokens):
        self.tokens=list(tokens);self.index={v:i for i,v in enumerate(self.tokens)}
        if self.tokens[:2]!=["<pad>","<cls>"] or len(self.index)!=len(self.tokens):
            raise ValueError("Invalid vocabulary")

    @classmethod
    def from_records(cls,records):
        vocabulary=set()
        for r in records:
            vocabulary.update(words(r["question"]))
            for c in r["candidates"]:vocabulary.update(words(c["text"]))
        return cls(["<pad>","<cls>",*sorted(vocabulary)])

    def encode(self,text,length=24):
        tokens=words(text)
        if not tokens:raise ValueError("Empty text")
        unknown=set(tokens)-self.index.keys()
        if unknown:raise ValueError(f"Words outside this prototype's learned vocabulary: {sorted(unknown)}")
        if len(tokens)>length:raise ValueError("Text exceeds the declared token budget; no silent truncation")
        return [self.index[t] for t in tokens]+[0]*(length-len(tokens))


class TextEncoder(nn.Module):
    def __init__(self,vocabulary,width,max_tokens=24):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,width,padding_idx=0)
        self.positions=nn.Parameter(torch.randn(1,max_tokens+1,width)*.02)
        self.block=nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(width)

    def forward(self,tokens):
        cls=torch.ones((len(tokens),1),dtype=torch.long,device=tokens.device)
        tokens=torch.cat([cls,tokens],dim=1)
        x=self.embedding(tokens)+self.positions[:,:tokens.shape[1]]
        x=self.block(x,src_key_padding_mask=tokens.eq(0))
        return self.norm(x[:,0])


class NativeDecisionModel(nn.Module):
    def __init__(self,vocabulary,width=128,layers=2,audio_initialization=None):
        super().__init__()
        base=AudioCNN(8)
        if audio_initialization:
            base.load_state_dict(load_file(str(audio_initialization)))
        # Drop the fixed eight-class head. Retain a learned acoustic representation,
        # then jointly fine-tune it along with image, text, fusion, and scoring.
        self.audio=nn.Sequential(base.features,base.classifier[0],base.classifier[1])
        self.audio_projection=nn.Linear(64,width)
        self.image=nn.Sequential(nn.Conv2d(3,24,3,2,1),nn.GroupNorm(4,24),nn.GELU(),
            nn.Conv2d(24,48,3,2,1),nn.GroupNorm(6,48),nn.GELU(),
            nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,width))
        self.text=TextEncoder(vocabulary,width)
        self.image_positions=nn.Parameter(torch.randn(1,4,width)*.02)
        self.roles=nn.Parameter(torch.randn(3,width)*.02)
        self.missing_image=nn.Parameter(torch.zeros(width));self.missing_audio=nn.Parameter(torch.zeros(width))
        self.fusion=nn.ModuleList([nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,activation="gelu",batch_first=True,norm_first=True) for _ in range(layers)])
        self.state_norm=nn.LayerNorm(width)
        self.read=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.score=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,1))

    def encode_observations(self,image,audio):
        # Fixed, label-independent quadrants are a deliberate inductive bias for
        # the generated 2x2 panels. They are not oracle target crops.
        patches=torch.stack([image[:,:,:64,:64],image[:,:,:64,64:],image[:,:,64:,:64],image[:,:,64:,64:]],dim=1)
        visual=self.image(patches.reshape(-1,3,64,64)).reshape(len(image),4,-1)
        acoustic=self.audio_projection(self.audio(audio))
        return visual,acoustic

    def decide(self,visual,acoustic,question,candidates,candidate_mask,modality_mask=None):
        if modality_mask is None:
            modality_mask=torch.ones((len(visual),2),dtype=torch.bool,device=visual.device)
        visual=torch.where(modality_mask[:,0,None,None],visual,self.missing_image[None,None,:])
        acoustic=torch.where(modality_mask[:,1,None],acoustic,self.missing_audio[None,:])
        q=self.text(question)
        state=torch.cat([visual+self.image_positions+self.roles[0],
                         (acoustic+self.roles[1])[:,None],(q+self.roles[2])[:,None]],dim=1)
        for block in self.fusion:state=block(state)
        state=self.state_norm(state)
        b,k,length=candidates.shape
        descriptions=self.text(candidates.reshape(b*k,length)).reshape(b,k,-1)
        evidence,_=self.read(descriptions,state,state,need_weights=False)
        logits=self.score(torch.cat([descriptions,evidence,descriptions*evidence],dim=-1)).squeeze(-1)
        return logits.masked_fill(~candidate_mask,-1e4)

    def forward(self,image,audio,question,candidates,candidate_mask,modality_mask=None):
        visual,acoustic=self.encode_observations(image,audio)
        return self.decide(visual,acoustic,question,candidates,candidate_mask,modality_mask)


def image_tensor(path):
    with Image.open(path) as image:
        image=image.convert("RGB").resize((128,128),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/127.5-1


def audio_tensor(path):
    audio=read_audio(path)
    audio=torch.nn.functional.pad(audio[:16000],(0,max(0,16000-len(audio))))
    return LogMel()(audio)


def encode_requests(requests,vocabulary):
    if not requests:raise ValueError("Supply at least one question")
    maximum=max(len(r["candidates"]) for r in requests)
    if maximum>32:raise ValueError("At most 32 candidates in this prototype")
    qs=[];cs=[];masks=[]
    for r in requests:
        options=r["candidates"]
        if len(options)<2:raise ValueError("At least two candidates are required")
        ids=[c["id"] for c in options];texts=[c["text"] for c in options]
        if not all(isinstance(i,str) and i for i in ids) or len(set(ids))!=len(ids):raise ValueError("Candidate IDs must be unique nonempty strings")
        if len({tuple(words(t)) for t in texts})!=len(texts):raise ValueError("Duplicate candidate meanings are not supported")
        qs.append(vocabulary.encode(r["question"],24))
        cs.append([vocabulary.encode(t,8) for t in texts]+[[0]*8]*(maximum-len(texts)))
        masks.append([True]*len(texts)+[False]*(maximum-len(texts)))
    return torch.tensor(qs),torch.tensor(cs),torch.tensor(masks)


def predict_joint(checkpoint,image_path,audio_path,requests,device="auto",threshold=0.):
    checkpoint=Path(checkpoint)
    if not 0<=threshold<=1:raise ValueError("Threshold must be in [0,1]")
    config=json.loads(checkpoint.with_name("config.json").read_text())
    vocabulary=Vocabulary(config["vocabulary"]);chosen=choose_device(device)
    model=NativeDecisionModel(len(vocabulary.tokens),config["width"],config["layers"]).to(chosen)
    model.load_state_dict(load_file(str(checkpoint)));model.eval()
    q,c,m=encode_requests(requests,vocabulary)
    image=image_tensor(image_path)[None].to(chosen);audio=audio_tensor(audio_path)[None].to(chosen)
    with torch.inference_mode():
        visual,acoustic=model.encode_observations(image,audio)
        modality=torch.ones((len(requests),2),dtype=torch.bool,device=chosen)
        if config.get("mode")=="audio_only":modality[:,0]=False
        elif config.get("mode")=="image_only":modality[:,1]=False
        logits=model.decide(visual.expand(len(requests),-1,-1),acoustic.expand(len(requests),-1),q.to(chosen),c.to(chosen),m.to(chosen),modality)
        probabilities=(logits/config["temperature"]).softmax(-1).cpu().numpy()
    answers=[]
    for r,p in zip(requests,probabilities):
        p=p[:len(r["candidates"])];ids=[c["id"] for c in r["candidates"]]
        best=int(p.argmax());ties=np.flatnonzero(np.isclose(p,p[best],atol=1e-7,rtol=0)).tolist()
        abstain=float(p[best])<threshold or len(ties)>1
        answers.append({"type":"choice","probabilities":dict(zip(ids,p.tolist())),
                        "prediction":ids[best] if len(ties)==1 else None,"decision":"abstain" if abstain else ids[best],
                        "ties":[ids[i] for i in ties] if len(ties)>1 else [],"confidence":float(p[best])})
    return {"answers":answers,"scope":config["scope"],"model":"native-decision-v1",
            "note":"Raw audio and pixels enter the network; no transcript or scene oracle is used."}
