"""Freeze broader train/development/confirmation groups and source omissions."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from mmso.backbone_study import digest
from mmso.frontier_evals import point_inside
from mmso.grounding_tasks import pointer_from_script

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/v4-training'


def rank(value): return hashlib.sha256(('miso-v4-grounding-training-v1:'+str(value)).encode()).hexdigest()
def read(path): return [json.loads(s) for s in path.read_text().splitlines()]
def pixel_hash(image):
    rgb=image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
def stored(payload,relative):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source media path')
    path=BASE/'media'/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(payload)
    elif path.read_bytes()!=payload:raise ValueError('Media payload changed')
    with Image.open(io.BytesIO(payload)) as image:size=list(image.size);pixels=pixel_hash(image)
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'pixel_sha256':pixels,'modality':'image'},size
def groups_first(pool,limit):
    groups=defaultdict(list)
    for row in pool:groups[row['group_id']].append(row)
    order=sorted(groups,key=rank);out=[]
    for group in groups:groups[group].sort(key=lambda r:rank(r['id']))
    while len(out)<limit and any(groups.values()):
        for group in order:
            if groups[group]:out.append(groups[group].pop(0))
            if len(out)==limit:break
    return out


def main():
    protocol=json.loads((ROOT/'evals/v4-grounding-training-protocol-v1.json').read_text());settings=protocol['sampling']
    source_hashes={};omissions=[];rows=[]
    # Protect all previously exposed visual audits from the new training corpus.
    protected_pixels=set()
    for path in ('data/v4-frontier/manifest.jsonl','evals/manifests/v4_fresh_screens_v2.jsonl'):
        for row in read(ROOT/path):
            for media in row['media']:
                if media['modality']=='image':
                    with Image.open(ROOT/media['path']) as image:protected_pixels.add(pixel_hash(image))
    # Reserve a new, original-coordinate benchmark before selecting training media.
    source=BASE/'sources/OS-Copilot--ScreenSpot-v2'
    archive=zipfile.ZipFile(source/'screenspotv2_image.zip')
    source_hashes[str((source/'screenspotv2_image.zip').relative_to(ROOT))]=digest(source/'screenspotv2_image.zip')
    for platform in ('desktop','web'):
        annotation=source/f'screenspot_{platform}_v2.json';source_hashes[str(annotation.relative_to(ROOT))]=digest(annotation)
        original=json.loads(annotation.read_text())
        used=set()
        for kind in ('text','icon'):
            taken=0
            for item in sorted((r for r in original if r['data_type']==kind),key=lambda r:rank(r['img_filename']+r['instruction'])):
                if item['img_filename'] in used:continue
                payload=archive.read('screenspotv2_image/'+item['img_filename'])
                media,size=stored(payload,'screenspot-v2/'+item['img_filename'])
                if media['pixel_sha256'] in protected_pixels:
                    omissions.append({'source':'screenspot_v2','id':item['img_filename'],'reason':'previously_exposed_pixels'});continue
                x,y,w,h=item['bbox'];box=[x,y,x+w,y+h]
                if not 0<=x<x+w<=size[0] or not 0<=y<y+h<=size[1]:raise ValueError('ScreenSpot-v2 xywh box out of bounds')
                point=[(x+w/2)/size[0],(y+h/2)/size[1]]
                used.add(item['img_filename']);protected_pixels.add(media['pixel_sha256'])
                rows.append({'id':'screenspot-v2:'+item['img_filename']+':'+kind,'split':'confirmation','task':'point',
                    'dataset':'screenspot_v2','stratum':platform+':'+kind,'group_id':media['pixel_sha256'],
                    'question':item['instruction'],'media':[media],'target_point':point,'target_bbox_xyxy':box,'image_size':size})
                taken+=1
                if taken==settings['screenspot_confirmation_per_platform_type']:break
            if taken!=settings['screenspot_confirmation_per_platform_type']:raise ValueError('ScreenSpot confirmation quota unavailable')
    # Old exposed screens are now explicitly development, never confirmation.
    for old in read(ROOT/'evals/manifests/v4_fresh_screens_v2.jsonl'):
        question=old['question'].split('Requested element: ',1)[1];box=old['target_bbox_xyxy'];w,h=old['image_size']
        rows.append({'id':'point-dev:'+old['id'],'split':'development','task':'point','dataset':'screenspot_pro',
            'stratum':old['stratum'],'group_id':old['group_id'],'question':question,'media':old['media'],
            'target_point':[(box[0]+box[2])/2/w,(box[1]+box[3])/2/h],'target_bbox_xyxy':box,'image_size':[w,h]})
    # OmniAct publisher data, using only literal single pointer actions.
    source=BASE/'sources/Writer--omniact';archive=zipfile.ZipFile(source/'data.zip');names=set(archive.namelist())
    for name in ('train.json','data.zip'):source_hashes[str((source/name).relative_to(ROOT))]=digest(source/name)
    canonical=defaultdict(list)
    def normalize(name):
        p=Path(name);return str(p.parent)+'/'+re.sub(r'[_-]','',p.stem.replace('_boxes',''))+p.suffix
    for name in names:canonical[normalize(name)].append(name)
    def resolve(name):
        if name in names:return name
        matches=canonical[normalize(name)]
        if len(matches)!=1:raise ValueError('missing_or_ambiguous_source_path')
        return matches[0]
    pool=[];media_cache={};box_cache={}
    for identity,item in json.loads((source/'train.json').read_text()).items():
        try:
            question,point=pointer_from_script(archive.read(resolve(item['task'])).decode())
            image_name=resolve(item['image']);box_name=resolve(item['box'])
            if image_name not in media_cache:media_cache[image_name]=stored(archive.read(image_name),'omniact/'+image_name.removeprefix('data/data/'))
            media,size=media_cache[image_name]
            if media['pixel_sha256'] in protected_pixels:raise ValueError('overlap_with_benchmark_pixels')
            if box_name not in box_cache:box_cache[box_name]=json.loads(archive.read(box_name))
            # Desktop annotations omit the web-only valid flag. Explicit web
            # invalid annotations remain excluded.
            boxes=[[*b['top_left'],*b['bottom_right']] for b in box_cache[box_name].values() if b.get('valid',1)==1]
            normalized=[point[0]/size[0],point[1]/size[1]]
            boxes=[b for b in boxes if point_inside(normalized,b,size) and b[2]>b[0] and b[3]>b[1]]
            if not boxes:raise ValueError('source_point_outside_valid_boxes')
            box=min(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            if not 0<=box[0]<box[2]<=size[0] or not 0<=box[1]<box[3]<=size[1]:raise ValueError('source_box_out_of_bounds')
            app='/'.join(Path(image_name).parts[2:4])
            bucket=int(rank(app)[:8],16)%10
            split='development' if bucket==0 else 'confirmation' if bucket==1 else 'train'
            pool.append({'id':'omniact:'+identity,'split':split,'task':'point','dataset':'omniact','stratum':app,
                'group_id':'omniact:'+app,'question':question,'media':[media],'target_point':normalized,
                'target_bbox_xyxy':box,'image_size':size,'source_paths':{'task':item['task'],'image':item['image'],'resolved_image':image_name,'box':box_name}})
        except (ValueError,KeyError) as error:omissions.append({'source':'omniact','id':identity,'reason':str(error)})
    for split in ('train','development','confirmation'):
        selected=groups_first([r for r in pool if r['split']==split],settings['omniact_'+split+'_max'])
        rows+=selected
    protected_pixels|={r['media'][0]['pixel_sha256'] for r in rows if r['dataset']=='omniact'}
    # Original ChartQA questions and answers, grouped by decoded image identity.
    source=BASE/'sources/HuggingFaceM4--ChartQA/data';seen_pixels=set(protected_pixels)
    for split,prefix in (('train','train-00000'),('development','val-'),('confirmation','test-')):
        path=next(source.glob(prefix+'*.parquet'));source_hashes[str(path.relative_to(ROOT))]=digest(path)
        candidates=[]
        for i,item in enumerate(pq.read_table(path).to_pylist()):
            payload=item['image']['bytes']
            with Image.open(io.BytesIO(payload)) as image:pixels=pixel_hash(image)
            if pixels in seen_pixels:continue
            candidates.append((rank(f'{split}:{i}'),i,item,pixels))
        count=0;used=set()
        for _,i,item,pixels in sorted(candidates):
            if pixels in used:continue
            media,size=stored(item['image']['bytes'],f'chartqa/{split}-{i}.image')
            rows.append({'id':f'chartqa:{split}:{i}','split':split,'task':'chart','dataset':'chartqa',
                'stratum':str(item['human_or_machine']),'group_id':'chart:'+pixels,'question':item['query'],
                'media':[media],'answers':item['label'],'image_size':size})
            used.add(pixels);count+=1
            if count==settings['chart_'+split]:break
        if count!=settings['chart_'+split]:raise ValueError('Chart quota unavailable')
        seen_pixels|=used
    # BoolQ shares no source article title between partitions.
    source=BASE/'sources/google--boolq/data';titles=set()
    for split,prefix in (('train','train-'),('development','validation-'),('confirmation','validation-')):
        path=next(source.glob(prefix+'*.parquet'));source_hashes[str(path.relative_to(ROOT))]=digest(path)
        candidates=pq.read_table(path).to_pylist();selected=[]
        for i,item in sorted(enumerate(candidates),key=lambda pair:rank(f'{prefix}:{pair[0]}')):
            title=' '.join(item['passage'].split())
            group=hashlib.sha256(title.casefold().encode()).hexdigest()
            # The publisher mirror has question/answer/passage, but no title field.
            if group in titles:continue
            titles.add(group);selected.append({'id':f'boolq:{prefix}:{i}','split':split,'task':'choice','dataset':'boolq',
                'stratum':str(item['answer']),'group_id':'passage:'+group,'question':'Passage: '+item['passage']+'\nQuestion: '+item['question'],
                'choices':['No','Yes'],'target':int(item['answer']),'media':[]})
            if len(selected)==settings['boolq_'+split]:break
        if len(selected)!=settings['boolq_'+split]:raise ValueError('BoolQ quota unavailable')
        rows+=selected
    # Existing raw-audio and joint training replay; never use audit labels.
    old=read(ROOT/'evals/manifests/v4_selection_v1.jsonl')
    for track in ('speech_intent','sound_events','joint_control'):
        for r in sorted((r for r in old if r['split']=='train' and r['track']==track),key=lambda r:rank(r['id']))[:settings['native_train_per_track']]:
            rows.append({**r,'task':'choice','dataset':'native_replay','stratum':track})
    public=read(ROOT/'data/v4-frontier/manifest.jsonl');quotas={'mmau':128,'mmstar':120,'mmlu_pro':112}
    for benchmark,limit in quotas.items():
        pools=defaultdict(list)
        for row in public:
            if row['benchmark']==benchmark:pools[row['stratum']].append(row)
        ordered=[]
        for key in pools:pools[key].sort(key=lambda r:rank(r['id']))
        while len(ordered)<limit:
            for key in sorted(pools):
                if pools[key]:ordered.append(pools[key].pop(0))
                if len(ordered)==limit:break
        for row in ordered:rows.append({**row,'split':'regression','task':'choice','dataset':benchmark})
    # Check group and decoded-pixel disjointness across train/dev/confirmation.
    groups={};pixels={}
    for row in rows:
        if row['split']=='regression':continue
        group=row['group_id'];split=row['split']
        if group in groups and groups[group]!=split:raise ValueError('Group leakage')
        groups[group]=split
        for media in row['media']:
            key=media.get('pixel_sha256',media['sha256'])
            if key in pixels and pixels[key]!=split:raise ValueError('Media leakage')
            pixels[key]=split
        if row['task']=='point':
            rounded=[round(v,6) for v in row['target_point']]
            if not point_inside(rounded,row['target_bbox_xyxy'],row['image_size']):raise ValueError('Unreachable label under output contract')
    path=BASE/'manifest.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if path.exists() and path.read_text()!=content:raise ValueError('Frozen training manifest changed')
    path.write_text(content)
    index=[{k:r[k] for k in ('id','split','task','dataset','stratum','group_id')}|{'case_sha256':hashlib.sha256(json.dumps(r,ensure_ascii=False).encode()).hexdigest(),'media':r['media']} for r in rows]
    (ROOT/'evals/manifests/v4_grounding_training_v1.index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    audit={'manifest_sha256':digest(path),'sources_sha256':source_hashes,'counts':dict(Counter(r['dataset']+':'+r['split'] for r in rows)),
        'omissions_by_reason':dict(Counter(r['reason'] for r in omissions)),'point_label_reachability':1.0,
        'no_cross_partition_groups_or_exact_media':True,'pretraining_overlap':'unknown','research_only':True,
        'boolq_grouping':'Exact normalized passage hash. This mirror has no title column; broader same-article overlap is not established.'}
    (ROOT/'evals/acquisition/v4_grounding_training_v1.json').write_text(json.dumps(audit,indent=2)+'\n')
    (BASE/'omissions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in omissions))
    print(json.dumps(audit['counts']),flush=True)


if __name__=='__main__':main()
