"""Model-independent scoring of complete prediction files."""
import numpy as np

from .artifacts import ROOT, read_manifest, sha256, validate_manifest
from .metrics import aligned_predictions, categorical_metrics, grounding_metrics, joint_metrics, multilabel_metrics


def score_files(manifest, predictions, split="test", field="probabilities", verify_media=True):
    rows=read_manifest(manifest)
    validate_manifest(rows,verify_media=verify_media)
    selected=[r for r in rows if r["split"]==split]
    if not selected:
        raise ValueError("Selected split is empty")
    kinds={r["kind"] for r in selected}
    if len(kinds)!=1:
        raise ValueError("Score task kinds separately")
    values=aligned_predictions(selected,read_manifest(predictions))
    kind=next(iter(kinds))
    if kind in {"categorical","multilabel"}:
        labels=selected[0]["labels"]
        if any(r["labels"]!=labels for r in selected):
            raise ValueError("Prediction columns require one consistent label schema")
        if kind=="categorical":
            metrics=categorical_metrics(np.array([labels.index(r["target"]) for r in selected]),[p[field] for p in values],labels)
        else:
            target=[[int(label in r["targets"]) for label in labels] for r in selected]
            metrics=multilabel_metrics(target,[p[field] for p in values])
    elif kind=="grounding":
        metrics=grounding_metrics([r["target_bbox_xyxy"] for r in selected],[p[field] for p in values])
    else:
        metrics=joint_metrics([r["acceptable_answers"] for r in selected],values)
    return {"kind":kind,"split":split,"manifest_sha256":sha256(manifest),"predictions_sha256":sha256(predictions),
            "media_verified":verify_media,"metrics":metrics}
