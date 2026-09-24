"""Versioned semisynthetic paired data with untouched speaker groups."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np

from .artifacts import ROOT, read_manifest, sha256, write_json, write_manifest
from .data import MINI_SHA256, audio_media, rank
from .joint_world import JOINT_TASKS, POSITIONS, TASKS, TEMPLATES, WORDS, answer, answer_space, make_panel, render_panel

MANIFEST = ROOT / "evals/manifests/joint_panels_v1.jsonl"


def seed_for(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8],16)


def questions_for_scene(scene, paraphrase=False):
    rng=np.random.default_rng(seed_for(scene["id"]+":questions"))
    records=[]
    for task in TASKS if not paraphrase else JOINT_TASKS:
        pos=int(rng.integers(0,4)); template=2 if paraphrase else int(rng.integers(0,2))
        question=TEMPLATES[task][template].format(position=POSITIONS[pos])
        values=answer_space(task);rng.shuffle(values)
        # IDs are deliberately opaque and never tokenized as semantic input.
        candidates=[{"id":f"option-{seed_for(scene['id']+task+v):08x}","text":v} for v in values]
        target=answer(scene["panel"],scene["audio_word"],task,pos)
        target_index=values.index(target)
        records.append({"id":scene["id"]+":"+task+(":paraphrase" if paraphrase else ""),
            "scene_id":scene["id"],"split":scene["split"],"slice":"paraphrase" if paraphrase else scene["slice"],
            "task":task,"question":question,"candidates":candidates,"target_index":target_index,
            "target_text":target,"target_id":candidates[target_index]["id"],"speaker":scene["speaker"],
            "family_id":scene["family_id"]})
    return records


def expand_scenes(scenes):
    records=[]
    for scene in scenes:
        records.extend(questions_for_scene(scene))
        if scene["split"]=="test" and scene["slice"]=="in_distribution":
            records.extend(questions_for_scene(scene,paraphrase=True))
    return records


def audit_joint(scenes, verify_media=True):
    speakers={};audio_hashes={};image_hashes={};seen_ids=set();groups={};counts=Counter()
    old=read_manifest(ROOT/"evals/manifests/speech_keywords.jsonl")
    previously_used={r["group_id"].split(":",1)[1] for r in old}
    for s in scenes:
        if s["id"] in seen_ids:raise ValueError("Duplicate scene ID")
        seen_ids.add(s["id"])
        for store,key in [(speakers,s["speaker"]),(groups,s["family_id"]),
                          (audio_hashes,s["audio"]["sha256"]),(image_hashes,s["image"]["sha256"])]:
            if key in store and store[key]!=s["split"]:raise ValueError("Cross-split speaker, family, or media leakage")
            store[key]=s["split"]
        if s["split"]!="train" and s["speaker"] in previously_used:
            raise ValueError("Evaluation speaker was exposed to earlier baseline training or evaluation")
        if verify_media:
            for m in [s["audio"],s["image"]]:
                if sha256(ROOT/m["path"])!=m["sha256"]:raise ValueError("Joint media changed")
        counts[s["split"]+":"+s["slice"]]+=1
    records=expand_scenes(scenes)
    support=defaultdict(Counter)
    for r in records:support[r["split"]+":"+r["task"]][r["target_text"]]+=1
    return {"scenes":len(scenes),"questions":len(records),"scenes_by_slice":dict(counts),
            "speakers_by_split":dict(Counter(speakers.values())),"unique_audio":len(audio_hashes),"unique_images":len(image_hashes),
            "evaluation_speakers_never_used_in_prior_baseline":True,"split_groups_disjoint":True,
            "target_support":dict(support),"media_verified":verify_media}


def prepare_joint():
    archive=ROOT/"data/downloads/mini_speech_commands.zip"
    if sha256(archive)!=MINI_SHA256:raise ValueError("Prepare pinned speech data first")
    old=read_manifest(ROOT/"evals/manifests/speech_keywords.jsonl")
    seen={r["group_id"].split(":",1)[1] for r in old}
    pools=defaultdict(list)
    for r in old:
        if r["split"]=="train":
            pools[("train",r["target"])].append({"audio":r["media"][0],"speaker":r["group_id"].split(":",1)[1]})
    with zipfile.ZipFile(archive) as z:
        eligible=defaultdict(list)
        for name in z.namelist():
            parts=name.split("/")
            if len(parts)!=3 or parts[0]!="mini_speech_commands" or not name.endswith(".wav") or parts[1] not in WORDS:continue
            speaker=parts[2].split("_nohash_")[0]
            if speaker in seen:continue
            bucket=int(rank("joint:"+speaker)[:8],16)%10
            split="dev" if bucket<3 else "calibration" if bucket<5 else "test"
            eligible[(split,parts[1])].append((name,speaker))
        for (split,word),items in eligible.items():
            # Full eligible pool; source grouping fixed before pairing or rendering.
            for name,speaker in sorted(items,key=lambda x:rank(x[0])):
                path=ROOT/"data/joint/audio"/word/Path(name).name;path.parent.mkdir(parents=True,exist_ok=True)
                payload=z.read(name)
                if path.exists() and path.read_bytes()!=payload:raise ValueError("Cached audio changed")
                path.write_bytes(payload)
                pools[(split,word)].append({"audio":audio_media(path),"speaker":speaker})
    scenes=[]
    configurations=[("train","in_distribution",2048),("dev","in_distribution",192),
                    ("calibration","in_distribution",128),("test","in_distribution",192),
                    ("test","compositional",192)]
    for split,slice_name,count in configurations:
        for i in range(count):
            word=WORDS[i%len(WORDS)];pool=pools[(split,word)]
            offset=24 if slice_name=="compositional" else 0
            source=pool[(i//len(WORDS)+offset)%len(pool)]
            identifier=f"joint-v1:{split}:{slice_name}:{i:05d}"
            panel=make_panel(seed_for(identifier),slice_name=="compositional")
            destination=ROOT/"data/joint/images"/(identifier.replace(":","-")+".png")
            destination.parent.mkdir(parents=True,exist_ok=True);render_panel(panel).save(destination)
            scenes.append({"id":identifier,"family_id":identifier,"split":split,"slice":slice_name,
                "speaker":source["speaker"],"audio_word":word,"audio":source["audio"],"panel":panel,
                "image":{"path":str(destination.relative_to(ROOT)),"sha256":sha256(destination),"modality":"image"}})
    # Position intervention: identical audio and icon/color associations, moved tiles.
    originals=[s for s in scenes if s["split"]=="test" and s["slice"]=="in_distribution"][:96]
    for original in originals:
        scene=json.loads(json.dumps(original));scene["id"]+="-moved";scene["slice"]="position_intervention"
        tiles=scene["panel"]["tiles"];scene["panel"]["tiles"]=tiles[1:]+tiles[:1]
        dest=ROOT/"data/joint/images"/(scene["id"].replace(":","-")+".png");render_panel(scene["panel"]).save(dest)
        scene["image"]={"path":str(dest.relative_to(ROOT)),"sha256":sha256(dest),"modality":"image"};scenes.append(scene)
    audit=audit_joint(scenes)
    if MANIFEST.exists() and read_manifest(MANIFEST)!=scenes:
        raise ValueError("Frozen paired manifest changed; version the experiment")
    write_manifest(MANIFEST,scenes)
    write_json(ROOT/"evals/acquisition/joint_panels_v1.json",{"dataset":"joint_panels_v1","manifest_sha256":sha256(MANIFEST),
        "source_audio_archive_sha256":MINI_SHA256,"source_baseline_manifest_sha256":sha256(ROOT/"evals/manifests/speech_keywords.jsonl"),
        "audit":audit,"scope":"Real human single-word speech paired with original generated 2x2 icon panels; not real application screenshots",
        "oracle":"mmso/joint_world.py; never called by prediction","question_manifest":"Deterministically expanded from scenes by joint_data.py",
        "held_compositions":"(word index + color index) mod 4 == 0; excluded from train/dev/calibration",
        "data_license":"Speech Commands source attribution applies to audio; panels are generated by this project"})
    print(json.dumps(audit),flush=True)
    return MANIFEST
