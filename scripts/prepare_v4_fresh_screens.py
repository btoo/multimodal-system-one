"""Freeze unused screenshot identities before selecting an input variant."""
from collections import Counter
import json
from pathlib import Path
import random
from urllib.parse import quote

from mmso.artifacts import sha256
from mmso.data import download, SCREEN_REVISION
from mmso.frontier_evals import stable_rank
from mmso.v4_iteration import REGIONS

ROOT = Path(__file__).resolve().parents[1]


def main():
    used_names, used_hashes = set(), set()
    for name in ('screen_grounding.jsonl', 'screen_grounding_v2.jsonl', 'v4_selection_v1.jsonl'):
        for line in (ROOT / 'evals/manifests' / name).read_text().splitlines():
            row = json.loads(line)
            for item in row.get('media', []):
                if item['modality'] == 'image':
                    used_names.add(Path(item['path']).name); used_hashes.add(item['sha256'])
    rows, availability, sources = [], {}, {}
    base = f'https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/images/'
    for annotation in sorted((ROOT / 'data/screens_v2/annotations').glob('*.json')):
        sources[str(annotation.relative_to(ROOT))] = sha256(annotation)
        records = json.loads(annotation.read_text())
        for kind in ('text', 'icon'):
            taken = 0
            for record in sorted((r for r in records if r['ui_type'] == kind), key=lambda r: stable_rank(r['id'])):
                name = record['img_filename']
                if name in used_names: continue
                path = ROOT / 'data/screens_v2/images' / name
                download(base + quote(name, safe='/'), path, max_bytes=25 << 20)
                fingerprint = sha256(path)
                if fingerprint in used_hashes: continue
                used_names.add(name); used_hashes.add(fingerprint)
                width, height = record['img_size']; x0,y0,x1,y1 = record['bbox']
                col = min(2, int(((x0+x1)/2)/width*3)); row_idx=min(2,int(((y0+y1)/2)/height*3))
                choices = REGIONS.copy(); random.Random(stable_rank(record['id'])).shuffle(choices)
                rows.append({'id': 'fresh-screen:' + record['id'], 'split': 'confirmation', 'benchmark': 'screenspot_pro_derived',
                    'stratum': annotation.stem + ':' + kind, 'group_id': fingerprint, 'application': annotation.stem,
                    'ui_type': kind, 'question': 'In which of nine equal regions of the screenshot is the center of the requested UI element? '
                    '(Three columns: left, center, right; three rows: top, middle, bottom.) Requested element: ' + record['instruction'],
                    'choices': choices, 'target': choices.index(REGIONS[row_idx*3+col]),
                    'target_bbox_xyxy': record['bbox'], 'image_size': [width,height],
                    'media': [{'path': str(path.relative_to(ROOT)), 'modality': 'image', 'sha256': fingerprint}]})
                taken += 1
                if taken == 8: break
            availability[annotation.stem + ':' + kind] = taken
        print(json.dumps({'application': annotation.stem, 'collected': len(rows)}), flush=True)
    path = ROOT / 'evals/manifests/v4_fresh_screens_v2.jsonl'
    content = ''.join(json.dumps(r) + '\n' for r in rows)
    if path.exists() and path.read_text() != content: raise ValueError('Frozen confirmation cannot be overwritten')
    path.write_text(content)
    audit = {'source_revision': SCREEN_REVISION, 'sources_sha256': sources, 'cases': len(rows),
        'manifest_sha256': sha256(path), 'availability': availability,
        'selection': 'Up to 8 per application/type; ascending fixed hash; unseen screenshot names and hashes across all previous MiSO screen manifests',
        'application_disjoint': False, 'prior_miso_image_overlap': 0, 'pretraining_overlap': 'unknown'}
    (ROOT / 'evals/acquisition/v4_fresh_screens_v2.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__': main()
