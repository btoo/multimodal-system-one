"""Bounded joint-model training, development selection, and later held-out evaluation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from safetensors.torch import load_file, save_file

from .artifacts import ROOT, provenance, read_manifest, sha256, write_json, write_manifest
from .audio import choose_device, synchronize
from .joint_data import MANIFEST, audit_joint, expand_scenes
from .joint_model import NativeDecisionModel, Vocabulary, audio_tensor, encode_requests, image_tensor
from .joint_world import JOINT_TASKS


class PairedCache:
    def __init__(self,scenes,vocabulary,split):
        self.scenes=[s for s in scenes if s["split"]==split]
        self.records=expand_scenes(self.scenes)
        self.scene_indices={s["id"]:i for i,s in enumerate(self.scenes)}
        self.index=torch.tensor([self.scene_indices[r["scene_id"]] for r in self.records])
        self.images=torch.stack([image_tensor(ROOT/s["image"]["path"]) for s in self.scenes])
        cache={}
        self.audio=[]
        for s in self.scenes:
            digest=s["audio"]["sha256"]
            if digest not in cache:cache[digest]=audio_tensor(ROOT/s["audio"]["path"])
            self.audio.append(cache[digest])
        self.audio=torch.stack(self.audio)
        self.question,self.candidates,self.mask=encode_requests(self.records,vocabulary)
        self.targets=torch.tensor([r["target_index"] for r in self.records])

    def batch(self,indices,device,mode="full"):
        scenes=self.index[indices]
        modality=torch.ones((len(indices),2),dtype=torch.bool)
        if mode=="audio_only":modality[:,0]=False
        elif mode=="image_only":modality[:,1]=False
        elif mode!="full":raise ValueError("Invalid modality control")
        inputs=[self.images[scenes],self.audio[scenes],self.question[indices],self.candidates[indices],self.mask[indices],modality]
        return [x.to(device) for x in inputs],self.targets[indices].to(device)


def decision_metrics(records,logits,temperature=1.):
    if not records or len(records)!=len(logits) or not np.isfinite(temperature) or temperature<=0:
        raise ValueError("Require complete predictions and positive finite temperature")
    grouped=defaultdict(list);all_rows=[]
    for record,z in zip(records,logits):
        z=np.asarray(z[:len(record["candidates"])],dtype=float)/temperature
        p=softmax(z);target=record["target_index"];pred=int(p.argmax())
        nll=float(logsumexp(z)-z[target]);brier=float(np.square(p-np.eye(len(p))[target]).sum())
        item={"correct":float(pred==target),"nll":nll,"normalized_nll":nll/np.log(len(p)),"brier":brier,
              "confidence":float(p.max()),"id":record["id"],"predicted_text":record["candidates"][pred]["text"],
              "target_text":record["target_text"],"probabilities":p.tolist(),"task":record["task"],"slice":record["slice"]}
        grouped[(record["slice"],record["task"])].append(item);all_rows.append(item)
    metrics={}
    for (slice_name,task),rows in grouped.items():
        confidence=np.array([r["confidence"] for r in rows]);correct=np.array([r["correct"] for r in rows])
        bins=np.minimum((confidence*10).astype(int),9);ece=0.;reliability=[]
        for b in range(10):
            mask=bins==b;n=int(mask.sum())
            acc=float(correct[mask].mean()) if n else None;c=float(confidence[mask].mean()) if n else None
            if n:ece+=n/len(rows)*abs(acc-c)
            reliability.append({"count":n,"accuracy":acc,"mean_confidence":c})
        metrics.setdefault(slice_name,{})[task]={"examples":len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ["correct","nll","normalized_nll","brier"]},"ece":ece,"reliability":reliability}
    summary={}
    for slice_name,tasks in metrics.items():
        joint=[v for k,v in tasks.items() if k in JOINT_TASKS]
        summary[slice_name]={"joint_macro_accuracy":float(np.mean([v["correct"] for v in joint])) if joint else None,
                             "joint_macro_normalized_nll":float(np.mean([v["normalized_nll"] for v in joint])) if joint else None,
                             "joint_examples":sum(v["examples"] for v in joint)}
    return {"by_slice":summary,"by_task":metrics},all_rows


@torch.inference_mode()
def evaluate_logits(model,data,device,mode="full",batch_size=64):
    model.eval();values=[]
    for indices in torch.arange(len(data.records)).split(batch_size):
        inputs,_=data.batch(indices,device,mode)
        values.extend(model(*inputs).cpu().numpy())
    return np.asarray(values)


def fit_joint_temperature(records,logits):
    def loss(log_t):
        return float(np.mean([(logsumexp(z[:len(r["candidates"])]/np.exp(log_t))-z[r["target_index"]]/np.exp(log_t))/np.log(len(r["candidates"]))
                             for r,z in zip(records,logits)]))
    result=minimize_scalar(loss,bounds=(-2.,2.),method="bounded")
    return float(np.exp(result.x)) if result.success and result.fun<loss(0.) else 1.


def train_joint(run_id,mode="full",device="auto",epochs=48,max_train_seconds=360,seed=20260924,width=128,layers=2,max_steps=None):
    report_dir=ROOT/"reports"/run_id;checkpoint_dir=ROOT/"artifacts"/run_id
    if report_dir.exists() or checkpoint_dir.exists():raise ValueError("Use a fresh immutable run ID")
    if mode not in {"full","audio_only","image_only"}:raise ValueError("Unknown modality condition")
    start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(seed)
    chosen=choose_device(device);scenes=read_manifest(MANIFEST);audit=audit_joint(scenes)
    vocabulary=Vocabulary.from_records(expand_scenes([s for s in scenes if s["split"]=="train"]))
    training=PairedCache(scenes,vocabulary,"train");development=PairedCache(scenes,vocabulary,"dev")
    initial=ROOT/"artifacts/speech-cnn-v1/model.safetensors"
    model=NativeDecisionModel(len(vocabulary.tokens),width,layers,initial).to(chosen)
    audio_params=list(model.audio.parameters());audio_ids={id(p) for p in audio_params}
    other=[p for p in model.parameters() if id(p) not in audio_ids]
    optimizer=torch.optim.AdamW([{"params":audio_params,"lr":1e-4},{"params":other,"lr":3e-4}],weight_decay=.01)
    generator=torch.Generator().manual_seed(seed);history=[];spent=0.;steps=0;best=float("inf");best_state=None;best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for indices in torch.randperm(len(training.records),generator=generator).split(64):
            if spent>=max_train_seconds or (max_steps is not None and steps>=max_steps):break
            synchronize(chosen);t=time.perf_counter()
            inputs,targets=training.batch(indices,chosen,mode)
            optimizer.zero_grad(set_to_none=True);logits=model(*inputs)
            loss=nn.functional.cross_entropy(logits,targets)
            if not torch.isfinite(loss):raise ValueError("Nonfinite joint loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();synchronize(chosen)
            spent+=time.perf_counter()-t;steps+=1;losses.append(float(loss.detach().cpu()))
        if not losses:break
        dev_logits=evaluate_logits(model,development,chosen,mode)
        metrics,_=decision_metrics(development.records,dev_logits)
        score=metrics["by_slice"]["in_distribution"]
        row={"epoch":epoch,"steps":steps,"train_seconds":spent,"train_loss":float(np.mean(losses)),**score}
        history.append(row);print(json.dumps({"run_id":run_id,**row}),flush=True)
        if score["joint_macro_normalized_nll"]<best:
            best=score["joint_macro_normalized_nll"];best_epoch=epoch
            best_state={k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items()}
    if best_state is None:raise ValueError("No model update completed")
    checkpoint_dir.mkdir(parents=True)
    checkpoint=checkpoint_dir/"model.safetensors";save_file(best_state,str(checkpoint))
    model.load_state_dict(best_state)
    dev_logits=evaluate_logits(model,development,chosen,mode);dev_metrics,dev_predictions=decision_metrics(development.records,dev_logits)
    config={"architecture":"native-decision-v1","width":width,"layers":layers,"vocabulary":vocabulary.tokens,
            "mode":mode,"seed":seed,"epochs_cap":epochs,"train_seconds_cap":max_train_seconds,"actual_steps":steps,
            "selected_epoch":best_epoch,"batch_size":64,"audio_initialization_sha256":sha256(initial),"temperature":1.,
            "scope":"Real one-word speech and generated 2x2 icon panels; limited learned vocabulary; no real-browser capability claim"}
    write_json(checkpoint_dir/"config.json",config)
    write_manifest(report_dir/"development-predictions.jsonl",dev_predictions)
    report={"run_id":run_id,"status":"development_complete","kind":"native_joint_training","configuration":config,
            "provenance":provenance(MANIFEST),"dataset_audit":audit,"parameters":sum(p.numel() for p in model.parameters()),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"checkpoint":str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256":sha256(checkpoint),"history":history,"development":dev_metrics,
            "total_wall_seconds":time.perf_counter()-start,"test_evaluated":False}
    write_json(report_dir/"training.json",report)
    return report


def smoke_joint(device="auto",steps=250):
    """Discarded small-batch learnability check using training data only."""
    torch.set_num_threads(4);torch.manual_seed(731);chosen=choose_device(device)
    scenes=[s for s in read_manifest(MANIFEST) if s["split"]=="train"][:8]
    vocabulary=Vocabulary.from_records(expand_scenes([s for s in read_manifest(MANIFEST) if s["split"]=="train"]))
    data=PairedCache(scenes,vocabulary,"train")
    model=NativeDecisionModel(len(vocabulary.tokens),audio_initialization=ROOT/"artifacts/speech-cnn-v1/model.safetensors").to(chosen)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    inputs,targets=data.batch(torch.arange(len(data.records)),chosen)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);logits=model(*inputs);loss=nn.functional.cross_entropy(logits,targets)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%50==0:
            history.append({"step":step,"loss":float(loss.detach().cpu()),"accuracy":float((logits.argmax(1)==targets).float().mean().cpu())})
    model.eval()
    with torch.inference_mode():accuracy=float((model(*inputs).argmax(1)==targets).float().mean().cpu())
    result={"training_only_examples":len(data.records),"steps":steps,"accuracy":accuracy,"history":history,"checkpoint_retained":False}
    print(json.dumps(result),flush=True)
    return result


def evaluate_joint(run_id,device="auto"):
    report_dir=ROOT/"reports"/run_id
    if (report_dir/"evaluation.json").exists():raise ValueError("Final evaluation already recorded; do not overwrite")
    training=json.loads((report_dir/"training.json").read_text());config=training["configuration"]
    chosen=choose_device(device);torch.set_num_threads(4)
    checkpoint=ROOT/training["checkpoint"]
    if sha256(checkpoint)!=training["checkpoint_sha256"]:raise ValueError("Checkpoint changed after nomination")
    scenes=read_manifest(MANIFEST);audit_joint(scenes)
    vocabulary=Vocabulary(config["vocabulary"])
    model=NativeDecisionModel(len(vocabulary.tokens),config["width"],config["layers"]).to(chosen)
    model.load_state_dict(load_file(str(checkpoint)));model.eval()
    calibration=PairedCache(scenes,vocabulary,"calibration")
    temperature=fit_joint_temperature(calibration.records,evaluate_logits(model,calibration,chosen,config["mode"]))
    test=PairedCache(scenes,vocabulary,"test")
    logits=evaluate_logits(model,test,chosen,config["mode"])
    raw,_=decision_metrics(test.records,logits);calibrated,predictions=decision_metrics(test.records,logits,temperature)
    # Structural checks use the frozen model and identical observations.
    indices=torch.arange(min(64,len(test.records)));inputs,_=test.batch(indices,chosen,config["mode"])
    with torch.inference_mode():
        original=model(*inputs)
        permutation=torch.arange(inputs[3].shape[1]-1,-1,-1,device=chosen)
        altered=inputs.copy();altered[3]=inputs[3][:,permutation];altered[4]=inputs[4][:,permutation]
        reordered=model(*altered)[:,permutation]
        delta=float((original-reordered).abs().max().cpu())
        one=[x[:1] for x in inputs];alone=model(*one)
        isolation=float((alone-original[:1]).abs().max().cpu())
    # Matched-input question pairs demonstrate that question semantics matter.
    by_id={r["id"]:p for r,p in zip(test.records,predictions)}
    contrasts={}
    for first,second in [("color","opposite_color"),("position","opposite_position"),("present","absent")]:
        eligible=both=0
        for scene in [s for s in scenes if s["split"]=="test" and s["slice"]=="in_distribution"]:
            a=by_id[scene["id"]+":"+first];b=by_id[scene["id"]+":"+second]
            if a["target_text"]!=b["target_text"]:
                eligible+=1;both+=int(a["correct"] and b["correct"])
        contrasts[first+"_vs_"+second]={"pairs_with_different_answers":eligible,"both_correct":both,"both_correct_rate":both/eligible if eligible else None}
    # Candidate subsets contain the correct option but otherwise vary. They are
    # a structural/easier-choice check, not evidence for new semantic concepts.
    subsets=[]
    selected=test.records[:96]
    for k in [2,3,4]:
        subset_rows=[];observed=[]
        for i,r in enumerate(selected):
            if len(r["candidates"])<=k:continue
            keep=[r["target_index"]]+[j for j in range(len(r["candidates"])) if j!=r["target_index"]][:k-1]
            sub=dict(r);sub["candidates"]=[r["candidates"][j] for j in keep];sub["target_index"]=0
            q,c,m=encode_requests([sub],vocabulary)
            scene_index=int(test.index[i])
            with torch.inference_mode():
                z=model(test.images[scene_index:scene_index+1].to(chosen),test.audio[scene_index:scene_index+1].to(chosen),q.to(chosen),c.to(chosen),m.to(chosen),inputs[5][:1]).cpu().numpy()[0]
            observed.append(z);subset_rows.append(sub)
        if subset_rows:
            metric,_=decision_metrics(subset_rows,observed,temperature)
            subsets.append({"candidates":k,"examples":len(subset_rows),"metrics":metric})
    # Warm full pipeline: local files through preprocessing and all neural components.
    from .joint_model import audio_tensor,image_tensor
    timings=[]
    for r in [test.records[i] for i in np.random.default_rng(55).choice(len(test.records),32,replace=False)]:
        scene=test.scenes[test.scene_indices[r["scene_id"]]]
        synchronize(chosen);t=time.perf_counter()
        image=image_tensor(ROOT/scene["image"]["path"])[None].to(chosen);audio=audio_tensor(ROOT/scene["audio"]["path"])[None].to(chosen)
        q,c,m=encode_requests([r],vocabulary)
        with torch.inference_mode():p=(model(image,audio,q.to(chosen),c.to(chosen),m.to(chosen),inputs[5][:1])/temperature).softmax(-1).cpu()
        synchronize(chosen);timings.append(time.perf_counter()-t)
    config["temperature"]=temperature;write_json(checkpoint.with_name("config.json"),config)
    write_manifest(report_dir/"predictions.jsonl",predictions)
    report={"run_id":run_id,"status":"held_out_evaluated","mode":config["mode"],"provenance":provenance(MANIFEST),
            "checkpoint_sha256":sha256(checkpoint),"temperature":temperature,"raw":raw,"calibrated":calibrated,
            "candidate_permutation_max_logit_difference":delta,"question_batch_isolation_max_logit_difference":isolation,
            "question_contrasts":contrasts,"candidate_subsets":subsets,
            "timing":{"warm_pipeline_p50_ms":float(np.median(timings)*1000),"warm_pipeline_p95_ms":float(np.quantile(timings,.95)*1000),"samples":32,
                      "includes":["file decode","log-mel","image preparation","text encoding","fusion","candidate scoring","softmax","device sync"],
                      "excludes":["listening duration","checkpoint load"],"streaming":False},
            "limitations":["Generated fixed-layout panels, not real application screenshots","Eight spoken words and a limited learned question/candidate vocabulary",
                           "One training seed; warm-started from the earlier acoustic encoder","Subset tests do not establish understanding unseen words or novel tasks"]}
    write_json(report_dir/"evaluation.json",report)
    print(json.dumps({"run_id":run_id,"held_out":calibrated["by_slice"],"candidate_permutation_delta":delta,"question_isolation_delta":isolation}),flush=True)
    return report
