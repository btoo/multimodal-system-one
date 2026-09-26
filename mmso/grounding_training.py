"""Coordinate/short-answer capability baselines and bounded mixed adaptation."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import save_file
import torch
from torch import nn

from .backbone_study import Backbone, decode_media, digest, move
from .backbone_adapters import DecisionLoRA, language_projection_names, replace, load_and_merge_adapter
from .frontier_evals import point_inside
from .grounding_tasks import safe_input, task_prompt, parse_point, chart_score


def infer_task(backbone, root, safe, protocol):
    policy=protocol['input_policy'];started=time.perf_counter()
    backbone.events=[]
    images,audios=decode_media(root,safe['media'],policy)
    inputs,_=backbone.prepare(task_prompt(safe),images,audios)
    length=inputs['input_ids'].shape[-1]
    if length>policy['max_input_tokens']:raise ValueError('Input token ceiling exceeded')
    inputs=move(inputs,'cuda');torch.cuda.synchronize();prepared=time.perf_counter()
    with torch.inference_mode():
        if safe['task']=='choice':
            output=backbone.forward(inputs)
            logits=output.logits[0,-1,backbone.token_ids[:len(safe['choices'])]].float()
            p=logits.softmax(-1).cpu().tolist()
            result={'probabilities':p,'choice':int(np.argmax(p)),'schema_valid':True,'output_tokens':0}
        else:
            maximum=protocol['generation'][safe['task']+'_max_new_tokens']
            kwargs=dict(max_new_tokens=maximum,do_sample=False,use_cache=True,return_dict_in_generate=True,output_hidden_states=False)
            if backbone.family=='minicpmo45':
                embeddings,_=backbone.model.get_vllm_embedding(inputs)
                embeddings=backbone.model.get_omni_embedding(inputs,embeddings,chunk_length=backbone.model.config.audio_chunk_length)
                terminators=[backbone.tokenizer.convert_tokens_to_ids(t) for t in backbone.model.terminators]
                output=backbone.model.llm.generate(inputs_embeds=embeddings,attention_mask=inputs['attention_mask'],pad_token_id=0,eos_token_id=terminators,**kwargs)
                ids=output.sequences[0]
            else:
                # The Omni publisher generation_config configures the Talker;
                # loading its Thinker alone does not inherit an EOS setting.
                eos=backbone.tokenizer.eos_token_id
                if eos is None:raise ValueError('Missing text end-of-answer token')
                output=backbone.model.generate(**inputs,eos_token_id=eos,pad_token_id=backbone.tokenizer.pad_token_id,**kwargs)
                ids=output.sequences[0,length:]
            raw=backbone.tokenizer.decode(ids,skip_special_tokens=True).strip()
            point=parse_point(raw,coordinate_scale=safe.get('coordinate_scale',1)) if safe['task']=='point' else None
            result={'text':raw,'point':point,'schema_valid':point is not None if safe['task']=='point' else bool(raw),
                'output_tokens':len(ids),'hit_output_limit':len(ids)>=maximum,
                'coordinate_scale':safe.get('coordinate_scale',1) if safe['task']=='point' else None,
                'last_token_id':int(ids[-1]) if len(ids) else None,
                'expected_stop_tokens':terminators if backbone.family=='minicpmo45' else [eos]}
    torch.cuda.synchronize();ended=time.perf_counter()
    return {'status':'ok',**result,'input_tokens':length,'timings':{'request_ms':(ended-started)*1000,'decode_and_processor_ms':(prepared-started)*1000,'model_and_output_ms':(ended-prepared)*1000}}


def score_result(row,result):
    if result['status']!='ok':return {'correct':False}
    if row['task']=='point':
        point=result['point']
        return {'correct':point_inside(point,row['target_bbox_xyxy'],row['image_size']),
            'normalized_center_distance':float(np.linalg.norm(np.asarray(point)-np.asarray(row['target_point']))) if point is not None else None}
    if row['task']=='chart':return {'correct':chart_score(result['text'],row['answers'])}
    return {'correct':result['choice']==row['target'],'nll':-float(np.log(max(result['probabilities'][row['target']],1e-12)))}


def summarize(rows,records):
    cases={r['id']:r for r in rows};groups=defaultdict(list)
    if len(records)!=len(cases) or {r['id'] for r in records}!=set(cases):raise ValueError('Incomplete result accounting')
    for result in records:groups[cases[result['id']]['dataset']].append(result)
    report={}
    for dataset,values in groups.items():
        strata=defaultdict(list);clusters=defaultdict(list)
        for r in values:
            case=cases[r['id']];strata[case['stratum']].append(int(r['correct']));clusters[case['group_id']].append(int(r['correct']))
        totals=np.asarray([(sum(v),len(v)) for v in clusters.values()])
        sample=np.random.default_rng(20260925).integers(0,len(totals),(2000,len(totals)));draws=totals[sample].sum(1)
        times=[r['timings']['request_ms'] for r in values if r['status']=='ok']
        report[dataset]={'cases':len(values),'correct':sum(r['correct'] for r in values),'accuracy':float(np.mean([r['correct'] for r in values])),
            'group_bootstrap_95':np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),'groups':len(clusters),
            'errors':sum(r['status']!='ok' for r in values),'schema_valid':sum(r.get('schema_valid',False) for r in values),
            'output_limit_hits':sum(r.get('hit_output_limit',False) for r in values),
            'latency_ms':{'median':float(np.median(times)),'p95':float(np.quantile(times,.95))} if times else None,
            'strata':{k:{'cases':len(v),'accuracy':float(np.mean(v))} for k,v in sorted(strata.items())}}
    return report


def evaluate(backbone,root,rows,protocol,output,*,prefix='',deadline=None):
    output.mkdir(parents=True,exist_ok=True);records=[]
    # Pure input-schema warmups before scoring, not model-quality preselection.
    for task in ('point','chart','choice'):
        row=next((r for r in rows if r['task']==task),None)
        if row is not None:infer_task(backbone,root,safe_input(row),protocol)
    for row in sorted(rows,key=lambda r:__import__('hashlib').sha256(r['id'].encode()).hexdigest()):
        if deadline is not None and time.perf_counter()>deadline:raise TimeoutError('Evaluation deadline reached; partial predictions retained')
        try:result=infer_task(backbone,root,safe_input(row),protocol)
        except Exception as error:
            result={'status':'error','error_type':type(error).__name__,'error':str(error)};torch.cuda.empty_cache()
        record={'id':row['id'],'dataset':row['dataset'],'split':row['split'],'task':row['task'],**result,**score_result(row,result)}
        records.append(record)
        with (output/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        if len(records)%50==0:print(json.dumps({'phase':prefix,'completed':len(records),'total':len(rows)}),flush=True)
    result={'cases':len(records),'metrics':summarize(rows,records),'predictions_sha256':digest(output/'predictions.jsonl')}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def teacher_forcing_inputs(backbone,inputs,answer):
    tokens=backbone.tokenizer.encode(answer,add_special_tokens=False)
    eos=backbone.tokenizer.convert_tokens_to_ids(backbone.model.terminators[0]) if backbone.family=='minicpmo45' else backbone.tokenizer.eos_token_id
    tokens.append(eos)
    device=inputs['input_ids'].device
    target=torch.tensor(tokens,device=device,dtype=torch.long)[None]
    inputs=dict(inputs);inputs['input_ids']=torch.cat([inputs['input_ids'],target[:,:-1]],dim=1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    if 'position_ids' in inputs:inputs['position_ids']=torch.arange(inputs['input_ids'].shape[-1],device=device)[None]
    return inputs,target


def train(backbone,root,rows,protocol,output):
    settings=protocol['training'];model=backbone.model
    torch.manual_seed(settings['seed']);torch.set_float32_matmul_precision('highest')
    wrappers={}
    for name in language_projection_names(model):
        wrapper=DecisionLoRA(model.get_submodule(name),settings['rank'],settings['alpha']);replace(model,name,wrapper);wrappers[name]=wrapper
    parameters=[p for layer in wrappers.values() for p in (layer.a,layer.b)]
    optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    checkpoint_model=getattr(model,'llm',model);checkpoint_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    for name in backbone.encoder_names:model.get_submodule(name).eval()
    pools={k:[] for k in settings['schedule']};queues={k:[] for k in pools}
    for row in rows:
        if row['split']!='train':continue
        group='native_replay' if row['dataset']=='native_replay' else 'boolq' if row['dataset']=='boolq' else row['task']
        pools[group].append(row)
    if any(not pool for pool in pools.values()):raise ValueError('Missing training task')
    rngs={k:random.Random(f"{settings['seed']}:{k}") for k in pools};start=time.perf_counter();seen=set();positive_gradient=False
    for step in range(settings['steps']):
        if time.perf_counter()-start>2200:raise TimeoutError('Training time ceiling reached; partial checkpoint is ineligible')
        group=settings['schedule'][step%len(settings['schedule'])]
        if not queues[group]:queues[group]=pools[group].copy();rngs[group].shuffle(queues[group])
        row=queues[group].pop();seen.add(row['id']);safe=safe_input(row)
        images,audios=decode_media(root,safe['media'],protocol['input_policy']);inputs,_=backbone.prepare(task_prompt(safe),images,audios);inputs=move(inputs,'cuda')
        optimizer.zero_grad(set_to_none=True)
        if row['task']=='choice':
            result=backbone.forward(inputs);logits=result.logits[:,-1,backbone.token_ids[:len(row['choices'])]].float()
            loss=nn.functional.cross_entropy(logits,torch.tensor([row['target']],device='cuda'))
            answer_tokens=1
        else:
            answer=json.dumps(dict(zip(('x','y'),[round(v,6) for v in row['target_point']])),separators=(',',':')) if row['task']=='point' else str(row['answers'][0])
            inputs,target=teacher_forcing_inputs(backbone,inputs,answer);answer_tokens=target.shape[-1]
            if inputs['input_ids'].shape[-1]>protocol['input_policy']['max_input_tokens']:raise ValueError('Training context exceeded')
            if backbone.family=='minicpmo45':result=model(data=inputs,use_cache=False,return_dict=True,logits_to_keep=answer_tokens)
            else:result=model(**inputs,use_cache=False,return_dict=True,logits_to_keep=answer_tokens)
            loss=nn.functional.cross_entropy(result.logits.float().reshape(-1,result.logits.shape[-1]),target.reshape(-1))
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,settings['gradient_clip_norm'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
        positive_gradient|=float(norm)>0;optimizer.step()
        record={'step':step+1,'id':row['id'],'split':'train','task':row['task'],'dataset':row['dataset'],'loss':float(loss.detach()),'gradient_norm':float(norm),'answer_tokens':answer_tokens}
        with (output/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if (step+1)%32==0:print(json.dumps({'training_step':step+1,'steps':settings['steps'],'loss':record['loss']}),flush=True)
    if not positive_gradient:raise ValueError('No training gradient')
    dest=output/'adapter';dest.mkdir()
    save_file({name+'.'+key:getattr(layer,key).detach().cpu().contiguous() for name,layer in wrappers.items() for key in ('a','b')},str(dest/'adapter.safetensors'))
    meta={'base_model':backbone.spec['id'],'revision':backbone.spec['revision'],'modules':list(wrappers),'rank':settings['rank'],'alpha':settings['alpha'],
        'steps':settings['steps'],'trainable_parameters':sum(p.numel() for p in parameters),'unique_training_cases_seen':len(seen),
        'training_seconds':time.perf_counter()-start,'gradient_observed':positive_gradient,'research_only':True,
        'protocol_sha256':digest(root/'evals/v4-grounding-training-protocol-v1.json'),'manifest_sha256':digest(root/'data/v4-training/manifest.jsonl'),
        'adapter_sha256':digest(dest/'adapter.safetensors')}
    (dest/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
    for name,wrapper in wrappers.items():replace(model,name,wrapper.merge())
    del optimizer,parameters,wrappers
    checkpoint_model.gradient_checkpointing_disable();model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    torch.cuda.empty_cache();return meta


def run(root,key,phase,output):
    started=time.perf_counter();torch.set_num_threads(4)
    protocol=json.loads((root/'evals/v4-grounding-training-protocol-v1.json').read_text())
    registry=json.loads((root/'evals/v4-candidates-v1.json').read_text());spec=next(s for s in registry['models'] if s['key']==key)
    rows=[json.loads(s) for s in (root/'data/v4-training/manifest.jsonl').read_text().splitlines()]
    acquisition=json.loads((root/'evals/acquisition/v4_grounding_training_v1.json').read_text())
    if acquisition['manifest_sha256']!=digest(root/'data/v4-training/manifest.jsonl'):raise ValueError('Manifest changed')
    model=Backbone(spec);load_seconds=time.perf_counter()-started
    selected=[r for r in rows if r['split'] in ('development','regression')]
    summary={'key':key,'phase':phase,'model':spec,'load_seconds':load_seconds,'gpu':torch.cuda.get_device_name(),
        'manifest_sha256':acquisition['manifest_sha256'],'protocol_sha256':digest(root/'evals/v4-grounding-training-protocol-v1.json')}
    if phase=='native-coordinate-reference':
        if key!='qwen3-30ba3b':raise ValueError('Native-coordinate control is Qwen only')
        reference=json.loads((root/'evals/v4-native-coordinate-reference-v1.json').read_text())
        summary['reference_protocol_sha256']=digest(root/'evals/v4-native-coordinate-reference-v1.json')
        selected=[{**r,'coordinate_scale':reference['coordinate_scale']} for r in rows if r['task']=='point' and r['split'] in ('development','confirmation')]
    if phase=='train':
        nomination=json.loads((root/'evals/v4-grounding-training-nomination-v1.json').read_text())
        if nomination['key']!=key:raise ValueError('Training candidate not nominated')
        summary['training']=train(model,root,rows,protocol,output)
    if phase=='adapted-development':
        nomination=json.loads((root/'evals/v4-grounding-training-nomination-v1.json').read_text())
        if nomination['key']!=key:raise ValueError('Training candidate not nominated')
        summary['adapter']=load_and_merge_adapter(model.model,root/'artifacts/v4-grounding-training'/key/'adapter',spec)
    if phase=='native-coordinate-reference':
        for split in ('development','confirmation'):
            summary[split]=evaluate(model,root,[r for r in selected if r['split']==split],protocol,output/split,prefix='native-coordinate-'+split,deadline=started+3300)
    elif phase=='confirmation':
        nomination=json.loads((root/'evals/v4-grounding-confirmation-nomination-v1.json').read_text())
        folder=root/'artifacts/v4-grounding-training'/key/'adapter'
        if nomination['key']!=key or digest(folder/'config.json')!=nomination['adapter_config_sha256']:raise ValueError('Confirmation candidate changed')
        selected=[r for r in rows if r['split']=='confirmation']
        summary['base']=evaluate(model,root,selected,protocol,output/'base',prefix='confirmation-base',deadline=started+3300)
        load_and_merge_adapter(model.model,folder,spec)
        summary['adapted']=evaluate(model,root,selected,protocol,output/'adapted',prefix='confirmation-adapted',deadline=started+3300)
    elif phase!='train':
        summary['evaluation']=evaluate(model,root,selected,protocol,output/'evaluation',prefix=phase,deadline=started+3300)
    summary.update(status='completed',function_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
