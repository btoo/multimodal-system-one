"""Bounded research jobs for broader evaluation and GUI input ablations."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch

from .backbone_study import Backbone, decode_media, digest, move, prompt_for
from .backbone_adapters import load_and_merge_adapter
from .backbone_diagnostics import trim_head
from .frontier_evals import model_input, summarize, point_inside


def load_model(root, adapted=True):
    registry = json.loads((root / 'evals/v4-candidates-v1.json').read_text())
    spec = next(m for m in registry['models'] if m['key'] == 'minicpmo45')
    model = Backbone(spec)
    if adapted:
        load_and_merge_adapter(model.model, root / 'artifacts/v4-adapters/minicpmo45/adapter', spec)
        model.load_readout(root / 'artifacts/v4-selection/minicpmo45-adapted')
    if model.readout is not None: raise ValueError('This iteration requires the selected label-logit readout')
    trim_head(model)
    return model


def run_frontier(root, phase, output):
    started = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(20260925)
    policy = json.loads((root / 'evals/v4-iteration-protocol-v2.json').read_text())['input_policy']
    path = root / 'data/v4-frontier/manifest.jsonl'
    acquisition = json.loads((root / 'evals/acquisition/v4_frontier_v1.json').read_text())
    if digest(path) != acquisition['manifest_sha256']: raise ValueError('Frozen manifest changed')
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for row in rows:
        for item in row['media']:
            if digest(root / item['path']) != item['sha256']: raise ValueError('Media hash mismatch')
    model = load_model(root, adapted=phase == 'frontier-adapted')
    load_seconds = time.perf_counter() - started
    # Warm each supported modality before timed evaluation. No labels are read.
    for benchmark in ('mmstar', 'mmau', 'mmlu_pro'):
        row = next(r for r in rows if r['benchmark'] == benchmark)
        model.infer(root, model_input(row), policy)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for row in sorted(rows, key=lambda r: __import__('hashlib').sha256(r['id'].encode()).hexdigest()):
        if time.perf_counter() - started > 3350: raise TimeoutError('Bounded runtime exhausted')
        try:
            result, _ = model.infer(root, model_input(row), policy)
            # Avoid reporting research feature-copy work as inference latency.
            result.pop('unconstrained_top_token', None)
        except Exception as error:
            result = {'status': 'error', 'error_type': type(error).__name__, 'error': str(error)}
            torch.cuda.empty_cache()
        record = {'id': row['id'], 'benchmark': row['benchmark'], **result}
        records.append(record)
        with (output / 'predictions.jsonl').open('a') as stream: stream.write(json.dumps(record, allow_nan=False) + '\n')
        if len(records) % 100 == 0:
            print(json.dumps({'phase': phase, 'completed': len(records), 'total': len(rows), 'last_status': result['status']}), flush=True)
    summary = {'status': 'completed', 'phase': phase, 'model': model.spec, 'cases': len(rows),
        'temperature': model.temperature, 'metrics': summarize(rows, records), 'load_seconds': load_seconds,
        'function_seconds': time.perf_counter() - started, 'gpu': torch.cuda.get_device_name(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'manifest_sha256': digest(path),
        'predictions_sha256': digest(output / 'predictions.jsonl'),
        'timing_boundary': 'File read, media decode/resample, CPU processor, host-to-device, all model encoders, language forward, candidate scores and CPU probability transfer. Warm weights; excludes network and queue.',
        'trained_on_this_suite': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


REGIONS = [f'{r} {c}' for r in ('top', 'middle', 'bottom') for c in ('left', 'center', 'right')]


def grid_overlay(image, choices):
    """Deterministic geometric overlay; input contains no target annotations."""
    if set(choices) != set(REGIONS): raise ValueError('Grid requires exactly nine named regions')
    image = image.copy()
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for i in (1, 2):
        draw.line([(width * i / 3, 0), (width * i / 3, height)], fill='#ef35ef', width=2)
        draw.line([(0, height * i / 3), (width, height * i / 3)], fill='#ef35ef', width=2)
    for idx, region in enumerate(REGIONS):
        x, y = (idx % 3) * width / 3 + 5, (idx // 3) * height / 3 + 5
        draw.rectangle([x, y, x + 25, y + 25], fill='#ffffff', outline='#111111', width=1)
        draw.text((x + 4, y + 2), chr(65 + choices.index(region)), fill='#111111', font_size=20)
    return image


def screen_predict(model, root, row, variant, policy):
    started = time.perf_counter()
    images, audios = decode_media(root, row['media'], {**policy, 'image_max_side': variant['image_max_side']})
    decoded = time.perf_counter()
    query = model_input(row)
    if variant['grid_overlay']:
        images = [grid_overlay(im, query['choices']) for im in images]
        query = {**query, 'question': query['question'] + ' The magenta lines divide the screenshot into nine equal regions. The small corner labels show each region\'s answer letter. Locate the requested element, then return that region\'s letter.'}
    inputs, _ = model.prepare(prompt_for(query), images, audios)
    if inputs['input_ids'].shape[-1] > policy['max_input_tokens']: raise ValueError('Token budget exceeded')
    processed = time.perf_counter()
    inputs = move(inputs, 'cuda'); torch.cuda.synchronize()
    transferred = time.perf_counter()
    with torch.inference_mode():
        output = model.forward(inputs)
        torch.cuda.synchronize(); forwarded = time.perf_counter()
        p = (output.logits[0, -1, :len(query['choices'])].float() / model.temperature).softmax(-1).cpu().tolist()
    torch.cuda.synchronize(); ended = time.perf_counter()
    return {'status': 'ok', 'probabilities': p, 'tokens': inputs['input_ids'].shape[-1],
        'timings': {'decode_ms': (decoded-started)*1000, 'processor_ms': (processed-decoded)*1000,
            'h2d_ms': (transferred-processed)*1000, 'forward_ms': (forwarded-transferred)*1000,
            'score_and_d2h_ms': (ended-forwarded)*1000, 'request_ms': (ended-started)*1000}}


def run_screens(root, output):
    torch.set_num_threads(4); torch.manual_seed(20260925)
    protocol = json.loads((root / 'evals/v4-iteration-protocol-v2.json').read_text())
    rows = [json.loads(s) for s in (root / 'evals/manifests/v4_selection_v1.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['split'] == 'development' and r['track'] == 'screen_region']
    model = load_model(root)
    variants = protocol['screen_ablation']['variants']
    for variant in variants: screen_predict(model, root, rows[0], variant, protocol['input_policy'])
    records = []
    for i, row in enumerate(rows):
        # Rotate ordering so one configuration does not always get warmer media.
        for variant in variants[i % len(variants):] + variants[:i % len(variants)]:
            result = screen_predict(model, root, row, variant, protocol['input_policy'])
            result.update(id=row['id'], variant=variant['name'], correct=int(np.argmax(result['probabilities'])) == row['target'])
            records.append(result)
            with (output / 'screens.jsonl').open('a') as stream: stream.write(json.dumps(result) + '\n')
    metrics = {}
    for variant in variants:
        values = [r for r in records if r['variant'] == variant['name']]
        metrics[variant['name']] = {'cases': len(values), 'correct': sum(r['correct'] for r in values),
            'accuracy': float(np.mean([r['correct'] for r in values])),
            'median_request_ms': float(np.median([r['timings']['request_ms'] for r in values])),
            'p95_request_ms': float(np.quantile([r['timings']['request_ms'] for r in values], .95))}
    result = {'status': 'completed', 'phase': 'screens', 'metrics': metrics,
        'selection_data': 'Previously exposed v1 development only; not a fresh confirmation result',
        'nominee': min(metrics, key=lambda k: (-metrics[k]['accuracy'], metrics[k]['median_request_ms']))}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_screen_confirmation(root, output):
    torch.set_num_threads(4); torch.manual_seed(20260925)
    protocol = json.loads((root / 'evals/v4-iteration-protocol-v2.json').read_text())
    nomination = json.loads((root / 'evals/v4-screen-nomination-v2.json').read_text())
    manifest = root / 'evals/manifests/v4_fresh_screens_v2.jsonl'
    if digest(manifest) != nomination['confirmation_manifest_sha256']: raise ValueError('Confirmation changed')
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    model = load_model(root)
    variant = next(v for v in protocol['screen_ablation']['variants'] if v['name'] == nomination['variant'])
    records = []
    for row in rows:
        result = screen_predict(model, root, model_input(row), variant, protocol['input_policy'])
        choice = int(np.argmax(result['probabilities']))
        grid_index = REGIONS.index(row['choices'][choice])
        point = [(grid_index % 3 + .5) / 3, (grid_index // 3 + .5) / 3]
        result.update(id=row['id'], point=point, region_correct=choice == row['target'],
            click_correct=point_inside(point, row['target_bbox_xyxy'], row['image_size']))
        records.append(result)
        with (output / 'predictions.jsonl').open('a') as stream: stream.write(json.dumps(result) + '\n')
    summary = {'status': 'completed', 'phase': 'screen-confirmation', 'variant': variant,
        'metrics': summarize(rows, records), 'region_accuracy': sum(r['region_correct'] for r in records)/len(rows),
        'point_inside_box_accuracy': sum(r['click_correct'] for r in records)/len(rows), 'cases': len(rows),
        'point_predictor': 'Center of the selected equal 3x3 grid cell. A weak localization control, not a dedicated point head.',
        'nomination_sha256': digest(root / 'evals/v4-screen-nomination-v2.json')}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
