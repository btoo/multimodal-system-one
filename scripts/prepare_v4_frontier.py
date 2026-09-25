"""Fetch pinned public evals; keep source questions/media in ignored data/."""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import soundfile as sf

from mmso.backbone_study import digest
from mmso.frontier_evals import mmstar_question, stable_rank

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    protocol = json.loads((ROOT / 'evals/v4-iteration-protocol-v2.json').read_text())
    base = ROOT / 'data/v4-frontier'
    base.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for name in ('mmstar', 'mmau', 'mmlu_pro'):
        spec = protocol['frontier_audit'][name]
        path = Path(hf_hub_download(spec['repo'], spec['file'], repo_type='dataset', revision=spec['revision'],
                    local_dir=base / 'sources' / spec['repo'].replace('/', '--')))
        sources[name] = {**spec, 'sha256': digest(path), 'bytes': path.stat().st_size}
        # Small batches avoid materializing a 1.2 GB audio parquet in memory.
        records = (r for b in pq.ParquetFile(path).iter_batches(batch_size=16) for r in b.to_pylist())
        if name == 'mmlu_pro':
            categories = defaultdict(list)
            for r in records: categories[r['category']].append(r)
            records = [r for cat in sorted(categories) for r in sorted(categories[cat], key=lambda r: stable_rank(r['question_id']))[:32]]
        for r in records:
            items = []
            if name == 'mmstar':
                identifier = str(r['index'])
                question, choices = mmstar_question(r['question'])
                target = ord(r['answer']) - 65
                stratum = r['category']
                payload = r['image'] if isinstance(r['image'], bytes) else r['image']['bytes']; suffix = '.image'
            elif name == 'mmau':
                meta = json.loads(r['other_attributes'])
                identifier = meta['id']; question = r['instruction']
                choices = [re.sub(r'^\([A-J]\)\s*', '', c) for c in r['choices']]
                target = r['choices'].index(r['answer']); stratum = meta['task']
                payload = r['context']['bytes']; suffix = '.wav'
            else:
                identifier = str(r['question_id']); question = r['question']; choices = r['options']
                target = r['answer_index']; stratum = r['category']
            if name != 'mmlu_pro':
                file = base / 'media' / name / (identifier + suffix)
                file.parent.mkdir(parents=True, exist_ok=True)
                if not file.exists() or file.read_bytes() != payload: file.write_bytes(payload)
                item = {'path': str(file.relative_to(ROOT)), 'sha256': digest(file),
                        'modality': 'audio' if name == 'mmau' else 'image'}
                if name == 'mmau':
                    info = sf.info(io.BytesIO(payload)); item['duration_seconds'] = info.duration
                items.append(item)
            if not 2 <= len(choices) <= 10 or not 0 <= target < len(choices):
                raise ValueError('Invalid original candidate/target contract')
            group = items[0]['sha256'] if items else stable_rank(question)
            rows.append({'id': name + ':' + identifier, 'benchmark': name, 'stratum': stratum,
                'group_id': group, 'split': 'public_audit', 'question': question, 'choices': choices,
                'target': target, 'media': items})
    counts = Counter(r['benchmark'] for r in rows)
    assert counts == {'mmstar': 1500, 'mmau': 1000, 'mmlu_pro': 448}, counts
    old = [json.loads(s) for s in (ROOT / 'evals/manifests/v4_selection_v1.jsonl').read_text().splitlines()]
    trained = {m['sha256'] for r in old if r['split'] == 'train' for m in r['media']}
    overlap = [r['id'] for r in rows if any(m['sha256'] in trained for m in r['media'])]
    if overlap: raise ValueError('Benchmark media overlaps MiSO training media: ' + str(overlap))
    path = base / 'manifest.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    # A reconstructible public index avoids redistributing source prompts or media.
    public = []
    for r in rows:
        public.append({k: r[k] for k in ('id', 'benchmark', 'stratum', 'group_id', 'split')} |
                      {'case_sha256': __import__('hashlib').sha256(json.dumps(r, ensure_ascii=False).encode()).hexdigest(),
                       'media': r['media']})
    index = ROOT / 'evals/manifests/v4_frontier_v1.index.jsonl'
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in public)
    if index.exists() and index.read_text() != content:
        raise ValueError('Frozen frontier index changed; do not overwrite')
    index.write_text(content)
    audit = {'sources': sources, 'cases': dict(counts), 'total': len(rows), 'manifest_sha256': digest(path),
        'index_sha256': digest(index), 'exact_media_overlap_with_miso_training': overlap,
        'publisher_pretraining_overlap': 'unknown', 'labels_used_for_training': False,
        'audio_over_120s': [r['id'] for r in rows if any(m.get('duration_seconds', 0) > 120 for m in r['media'])]}
    (ROOT / 'evals/acquisition/v4_frontier_v1.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'cases': dict(counts), 'total': len(rows), 'audio_over_120s': len(audit['audio_over_120s'])}))


if __name__ == '__main__': prepare()
