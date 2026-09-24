"""Honest screen controls: center point, CLIP grid, and oracle-crop type classification."""
from __future__ import annotations

import time
import numpy as np
from PIL import Image
import torch

from .artifacts import ROOT, local_media_path, provenance, read_manifest, sha256, validate_manifest, write_json, write_manifest
from .audio import choose_device, synchronize
from .metrics import categorical_metrics, grounding_metrics, point_hit

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def grid_regions(width, height, columns=6, rows=4):
    if min(width,height,columns,rows) < 1 or columns > width or rows > height:
        raise ValueError("Grid and image dimensions must be positive and nonempty")
    return [(round(c*width/columns),round(r*height/rows),round((c+1)*width/columns),round((r+1)*height/rows))
            for r in range(rows) for c in range(columns)]


def region_center(region):
    x0,y0,x1,y1 = region
    return [(x0+x1)/2,(y0+y1)/2]


class ClipControl:
    def __init__(self, device="auto"):
        from huggingface_hub import snapshot_download
        from transformers import CLIPModel, CLIPProcessor
        self.device = choose_device(device)
        self.local_path = snapshot_download(MODEL_ID,revision=MODEL_REVISION,
            local_dir=ROOT / "data/models/clip-vit-base-patch32",
            allow_patterns=["*.json","merges.txt","vocab.json","pytorch_model.bin","README.md"])
        self.processor = CLIPProcessor.from_pretrained(self.local_path,local_files_only=True)
        # Official pinned weights; weights_only loader, no remote model code.
        self.model = CLIPModel.from_pretrained(self.local_path,local_files_only=True,weights_only=True).to(self.device).eval()

    @torch.inference_mode()
    def score(self, images, texts):
        inputs=self.processor(images=images,text=texts,return_tensors="pt",padding=True,truncation=True,max_length=77)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        return self.model(**inputs).logits_per_image.detach().cpu()

    def ground(self, image, instruction):
        """Receives no target annotation; its fixed candidate grid is independent of labels."""
        regions=grid_regions(*image.size)
        scores=[]
        for start in range(0,len(regions),8):
            crops=[image.crop(box) for box in regions[start:start+8]]
            scores.extend(self.score(crops,[instruction]).flatten().tolist())
        best=int(np.argmax(scores))
        return region_center(regions[best]),regions,scores


def run_screen(run_id, device="auto"):
    report_dir=ROOT / "reports" / run_id
    if report_dir.exists():
        raise ValueError("Run IDs are immutable; select a new ID")
    start=time.perf_counter();torch.set_num_threads(4)
    manifest=ROOT / "evals/manifests/screen_grounding.jsonl"
    rows=read_manifest(manifest);audit=validate_manifest(rows)
    baseline=ClipControl(device)
    load_seconds=time.perf_counter()-start
    predictions=[];types=[];targets=[];latencies=[];coverage=[]
    # Explicit inference warmup, outside reported per-example timings.
    baseline.score([Image.new("RGB",(224,224))],["a computer interface"])
    for index,r in enumerate(rows):
        synchronize(baseline.device);t=time.perf_counter()
        with Image.open(local_media_path(ROOT,r["media"][0]["path"])) as im:
            image=im.convert("RGB")
        point,regions,scores=baseline.ground(image,r["instruction"])
        synchronize(baseline.device);elapsed=time.perf_counter()-t
        latencies.append(elapsed)
        center=[image.width/2,image.height/2]
        box=r["target_bbox_xyxy"]
        coverage.append(any(point_hit(region_center(b),box) for b in regions))
        # Separate, explicitly oracle-crop classification task; not grounding evidence.
        crop=image.crop(tuple(box))
        crop_p=baseline.score([crop],["a text label in a computer interface","a graphical icon in a computer interface"]).softmax(-1).numpy()[0]
        types.append(crop_p);targets.append(["text","icon"].index(r["ui_type"]))
        predictions.append({"id":r["id"],"application":r["application"],"ui_type":r["ui_type"],
            "image_size":r["image_size"],"target_bbox_xyxy":box,"center_point":center,"clip_grid_point":point,
            "center_hit":point_hit(center,box),"clip_grid_hit":point_hit(point,box),
            "grid_has_point_in_target":coverage[-1],"oracle_crop_type_probabilities":crop_p.tolist(),
            "grounding_pipeline_seconds":elapsed})
        print(f"Screen control {index+1}/{len(rows)}",flush=True)
    boxes=[r["target_bbox_xyxy"] for r in rows]
    per_app={}
    for app in sorted({r["application"] for r in rows}):
        sub=[p for p in predictions if p["application"]==app]
        per_app[app]={"examples":len(sub),"center_hits":sum(p["center_hit"] for p in sub),"clip_grid_hits":sum(p["clip_grid_hit"] for p in sub)}
    write_manifest(report_dir / "predictions.jsonl",predictions)
    report={"run_id":run_id,"dataset":"screen_grounding","status":"completed","kind":"screen_controls",
        "scope":"24-case stratified implementation probe; no training or tuning on this slice",
        "provenance":provenance(manifest),"dataset_audit":audit,
        "model":{"id":MODEL_ID,"revision":MODEL_REVISION,"weights_sha256":sha256(ROOT / "data/models/clip-vit-base-patch32/pytorch_model.bin"),
                 "parameters":sum(p.numel() for p in baseline.model.parameters()),"trainable_parameters":0,"device":str(baseline.device)},
        "center_point":grounding_metrics(boxes,[p["center_point"] for p in predictions]),
        "clip_grid":grounding_metrics(boxes,[p["clip_grid_point"] for p in predictions]),
        "grid_point_coverage_ceiling":{"examples":len(rows),"hits":sum(coverage),"rate":float(np.mean(coverage))},
        "oracle_crop_type":categorical_metrics(np.array(targets),types,["text","icon"]),"per_application":per_app,
        "configuration":{"grid_columns":6,"grid_rows":4,"image_batch_size":8,"output_point":"center of highest scoring fixed tile",
                         "crop_type_task":"ground truth crop supplied, classify text versus icon; separate diagnostic"},
        "timing":{"setup_including_download_seconds":load_seconds,"total_wall_seconds":time.perf_counter()-start,
            "grounding_pipeline_p50_ms":float(np.median(latencies)*1000),"grounding_pipeline_p95_ms":float(np.quantile(latencies,.95)*1000),
            "includes":["image read/decode","grid crops","CLIP preprocessing","model scoring","argmax","device synchronization"],
            "excludes":["model download/load","separate oracle-crop diagnostic"],"cold_process_latency_ms":None},
        "limitations":["Coarse grid centers cannot hit most small controls even with perfect tile ranking",
                        "CLIP is a generic pretrained similarity model, not a trained GUI grounder",
                        "Oracle-crop type accuracy is conditional image classification, not complete grounding",
                        "Only six applications and 24 cases; not the full ScreenSpot-Pro benchmark",
                        "No speech, browser execution, or joint-model evidence"]}
    write_json(report_dir / "report.json",report)
    print({"run_id":run_id,"center":report["center_point"],"clip_grid":report["clip_grid"],
           "oracle_crop_accuracy":report["oracle_crop_type"]["accuracy"]},flush=True)
    return report
