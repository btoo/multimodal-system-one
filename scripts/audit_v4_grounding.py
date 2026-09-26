"""Audit the scoring action space without new model calls or changing results."""
from pathlib import Path
import json

from PIL import Image

from mmso.backbone_study import digest
from mmso.frontier_evals import point_action_coverage

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / 'evals/manifests/v4_fresh_screens_v2.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    points = [((c+.5)/3, (r+.5)/3) for r in range(3) for c in range(3)]
    mismatches = []
    for row in rows:
        with Image.open(ROOT / row['media'][0]['path']) as image:
            if list(image.size) != row['image_size']: mismatches.append(row['id'])
    if mismatches: raise ValueError('Source image dimensions disagree with annotations')
    result = {**point_action_coverage(rows, points), 'manifest_sha256': digest(manifest),
        'source_image_dimension_mismatches': mismatches, 'new_model_inference': False,
        'correction': 'The earlier 0/117 grid-center result was guaranteed by the restricted output space. It is not evidence that the model lacks vision or cannot ground coordinates. Precise grounding remains unmeasured.',
        'preserved_valid_result': '60/117 coarse region classifications; this remains a different task from exact GUI grounding.',
        'original_predictions_preserved': True}
    path = ROOT / 'reports/v4-iteration-v2/grounding-action-space-audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('cases', 'reachable_target_count', 'oracle_accuracy_ceiling', 'can_discriminate_model_quality')}))


if __name__ == '__main__': main()
