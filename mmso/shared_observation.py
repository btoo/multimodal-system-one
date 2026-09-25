"""MiniCPM decision requests which encode one observation for many questions.

This adapter is deliberately model-specific: it uses the pinned publisher's
processor to expand placeholders and validates that every media token belongs
to the common prefix. Independent causal branches prevent cross-question input
leakage. It creates no persistent cross-request media cache.
"""
from __future__ import annotations

from collections import Counter
import json
import time

import numpy as np
import torch

from .backbone_study import decode_media, move, prompt_for
from .parallel_decisions import branch_plan, common_prefix, diagnostic_questions


def render_question(model, query, images, audios):
    content = '\n'.join([*('<image>./</image>' for _ in images), *('<audio>./</audio>' for _ in audios), prompt_for(query)])
    return model.tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, use_tts_template=bool(audios), enable_thinking=False)


def shared_inputs(model, images, audios, queries, policy):
    if model.family != 'minicpmo45' or model.readout is not None or not model.candidate_only:
        raise ValueError('Pinned MiniCPM label-logit adapter with reduced head required')
    if not queries or len(queries) > 16 or any(not 2 <= len(q['choices']) <= 10 for q in queries):
        raise ValueError('Expected 1-16 questions, each with 2-10 choices')
    processor = model.processor
    image_inputs = processor.process_image([images], max_slice_nums=9, return_tensors='pt')
    audio_features, audio_lens, audio_placeholders = processor.audio_feature_extract([audios], [[0] * len(audios)], False, 16000)
    prepared = []
    for query in queries:
        text = render_question(model, query, images, audios)
        data = processor._convert_omni_to_inputs(image_inputs, audio_placeholders, [text], max_slice_nums=9,
                    use_image_id=True, max_length=None, return_tensors='pt')
        data.pop('image_sizes', None)
        data.update(audio_features=audio_features, audio_feature_lens=audio_lens)
        length = data['input_ids'].shape[-1]
        if length > policy['max_input_tokens']: raise ValueError('Individual question exceeds input budget')
        data['position_ids'] = torch.arange(length)[None]
        prepared.append(data)
    sequences = [d['input_ids'][0] for d in prepared]
    prefix_length = min(common_prefix(sequences), min(len(s) for s in sequences) - 1)
    for data in prepared:
        for name in ('image_bound', 'audio_bounds', 'spk_bounds'):
            for bounds in data.get(name, []):
                if len(bounds) and int(bounds[:, 1].max()) > prefix_length:
                    raise ValueError('Media or speaker embedding extends beyond the shared prefix')
        for name in ('image_bound', 'audio_bounds', 'spk_bounds'):
            for a, b in zip(data.get(name, []), prepared[0].get(name, [])):
                if not torch.equal(a, b): raise ValueError('Question-dependent media bounds')
    return prepared, prefix_length


def predict_shared(model, root, media, queries, policy, *, packed=False):
    """Complete warm request from local files to CPU probability vectors."""
    started = time.perf_counter()
    images, audios = decode_media(root, media, policy)
    prepared, prefix_length = shared_inputs(model, images, audios, queries, policy)
    prepared_at = time.perf_counter()
    first = move(prepared[0], 'cuda')
    with torch.inference_mode():
        encoded, _ = model.model.get_vllm_embedding(first)
        encoded = model.model.get_omni_embedding(first, encoded, chunk_length=model.model.config.audio_chunk_length)
        sequences = [d['input_ids'][0] for d in prepared]
        prefix = encoded[:, :prefix_length]
        embed = model.model.llm.model.embed_tokens
        scale = getattr(model.model.llm.config, 'scale_emb', 1)
        suffixes = [embed(s[prefix_length:].to('cuda')[None]) * scale for s in sequences]
        torch.cuda.synchronize(); encoded_at = time.perf_counter()
        if packed:
            total = prefix_length + sum(len(s) - prefix_length for s in sequences)
            if total > policy['max_input_tokens']: raise ValueError('Packed request exceeds input budget')
            plan = branch_plan(sequences, prefix_length, device='cuda', dtype=encoded.dtype)
            result = model.model.llm(inputs_embeds=torch.cat([prefix, *suffixes], dim=1),
                position_ids=plan['position_ids'], attention_mask=plan['attention_mask'], use_cache=False,
                return_dict=True, logits_to_keep=plan['end_positions'])
            logits = [result.logits[0, j] for j in range(len(queries))]
        else:
            logits = []
            for data, suffix in zip(prepared, suffixes):
                result = model.model.llm(inputs_embeds=torch.cat([prefix, suffix], dim=1),
                    position_ids=data['position_ids'].to('cuda'), use_cache=False, return_dict=True, logits_to_keep=1)
                logits.append(result.logits[0, -1])
            total = sum(len(s) for s in sequences)
        probabilities = [(value[:len(q['choices'])].float() / model.temperature).softmax(-1).cpu().numpy()
                         for value, q in zip(logits, queries)]
    torch.cuda.synchronize()
    ended = time.perf_counter()
    return {'probabilities': probabilities, 'request_ms': (ended-started)*1000,
        'decode_and_processor_ms': (prepared_at-started)*1000, 'media_embedding_ms': (encoded_at-prepared_at)*1000,
        'language_and_output_ms': (ended-encoded_at)*1000, 'shared_prefix_tokens': prefix_length, 'language_tokens': total}


def independent(model, root, media, queries, policy):
    started = time.perf_counter()
    values = []
    with torch.inference_mode():
        for query in queries:
            images, audios = decode_media(root, media, policy)
            inputs, _ = model.prepare(prompt_for(query), images, audios)
            if inputs['input_ids'].shape[-1] > policy['max_input_tokens']: raise ValueError('Token budget exceeded')
            result = model.forward(move(inputs, 'cuda'))
            values.append((result.logits[0, -1, :len(query['choices'])].float() / model.temperature).softmax(-1).cpu().numpy())
    torch.cuda.synchronize()
    return {'probabilities': values, 'request_ms': (time.perf_counter()-started)*1000}


def run_full_pipeline_probe(root, output):
    from .v4_iteration import load_model
    torch.set_num_threads(4)
    protocol = json.loads((root / 'evals/v4-iteration-protocol-v2.json').read_text())
    settings, policy = protocol['shared_observation_speed'], protocol['input_policy']
    rows = [json.loads(s) for s in (root / 'evals/manifests/v4_selection_v1.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['split'] == 'development' and r['track'] == 'joint_control'][:settings['observations']]
    model = load_model(root)
    counts, handles = Counter(), []
    for name, module in model.model.named_modules():
        if name in model.encoder_names:
            handles.append(module.register_forward_pre_hook(lambda _m, _a, name=name: counts.update([name])))
    records = []
    for row in rows:
        for count in settings['questions']:
            queries = diagnostic_questions()[:count]
            # Verify the optimized processor emits exactly the original token IDs.
            images, audios = decode_media(root, row['media'], policy)
            compiled, _ = shared_inputs(model, images, audios, queries, policy)
            for query, data in zip(queries, compiled):
                original, _ = model.prepare(prompt_for(query), images, audios)
                if not torch.equal(data['input_ids'], original['input_ids']): raise ValueError('Compiled question token mismatch')
            paths = {
                'independent': lambda: independent(model, root, row['media'], queries, policy),
                'serial_shared_media': lambda: predict_shared(model, root, row['media'], queries, policy, packed=False),
                'packed_shared_media': lambda: predict_shared(model, root, row['media'], queries, policy, packed=True),
            }
            timings, values, invocations = {}, {}, {}
            for name, run in paths.items():
                counts.clear(); values[name] = run(); invocations[name] = dict(counts); timings[name] = []
            for repeat in range(settings['repeats']):
                names = list(paths); names = names[repeat % 3:] + names[:repeat % 3]
                for name in names:
                    torch.cuda.synchronize()
                    timings[name].append(paths[name]() ['request_ms'])
            reference = values['independent']['probabilities']
            comparisons = {}
            for name in ('serial_shared_media', 'packed_shared_media'):
                actual = values[name]['probabilities']
                comparisons[name] = {'argmax_agreement': sum(int(a.argmax()) == int(b.argmax()) for a,b in zip(reference, actual)),
                    'max_probability_delta': max(float(np.max(np.abs(a-b))) for a,b in zip(reference, actual)),
                    'probabilities': [p.tolist() for p in actual]}
            reverse = predict_shared(model, root, row['media'], list(reversed(queries)), policy, packed=True)['probabilities']
            reorder = max(float(np.max(np.abs(a-b))) for a,b in zip(values['packed_shared_media']['probabilities'], reversed(reverse)))
            record = {'id': row['id'], 'questions': count, 'timings_ms': timings,
                'encoder_invocations': invocations, 'comparison': comparisons,
                'reference_probabilities': [p.tolist() for p in reference], 'reorder_max_probability_delta': reorder,
                'shared_prefix_tokens': values['packed_shared_media']['shared_prefix_tokens'],
                'packed_language_tokens': values['packed_shared_media']['language_tokens'],
                'packed_stage_ms': {k:v for k,v in values['packed_shared_media'].items() if k.endswith('_ms')}}
            records.append(record)
            with (output / 'shared-observation.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
            print(json.dumps({'id': row['id'], 'questions': count, 'speedup': np.median(timings['independent'])/np.median(timings['packed_shared_media'])}), flush=True)
    for handle in handles: handle.remove()
    result = {'status': 'completed', 'phase': 'shared-observation', 'measurements': len(records),
        'observations': len(rows), 'gpu': torch.cuda.get_device_name(),
        'timing_boundary': 'Local file open through returned CPU probabilities; includes decode, resampling, all preprocessing, question compilation, media encoders, branch mask, language and scoring. Excludes network, queue and cold model load.',
        'new_quality_result': False}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
