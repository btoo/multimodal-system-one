"""Verify cloud evidence; optional local model and downloaded-checkpoint checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mmso.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='modal-l4-pilot-v3')
    parser.add_argument('--recompute-local', action='store_true')
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args()
    directory = ROOT / 'reports' / args.run_id
    result = json.loads((directory / 'result.json').read_text())
    bundle = json.loads((directory / 'bundle.json').read_text())
    reference = json.loads((directory / 'reference.json').read_text())
    resumed = json.loads((directory / 'resume.json').read_text())
    assert result['status'] == 'cloud_portability_and_resume_verified'
    assert result['accuracy_trial'] is False
    assert reference['process_id'] != resumed['process_id']
    source = bundle['source_commit']
    assert result['source_commit'] == reference['source_commit'] == resumed['source_commit'] == source
    assert reference['bundle_sha256'] == resumed['bundle_sha256'] == sha256(directory / 'bundle.json')
    checked = 0
    for relative, expected in bundle['files'].items():
        if relative == 'pilot-scenes.jsonl' or relative.startswith('data/'):
            continue
        content = subprocess.check_output(['git', 'show', f'{source}:{relative}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected, relative
        checked += 1
    raw_scenes = subprocess.check_output(['git', 'show', f'{source}:evals/manifests/joint_panels_v1.jsonl'], cwd=ROOT, text=True)
    scenes = [json.loads(line) for line in raw_scenes.splitlines() if json.loads(line)['split'] == 'train'][:64]
    scene_bytes = ''.join(json.dumps(s, sort_keys=True, allow_nan=False) + '\n' for s in scenes).encode()
    assert hashlib.sha256(scene_bytes).hexdigest() == bundle['files']['pilot-scenes.jsonl']
    inf = reference['inference']
    cpu, gpu = np.array(inf['cpu_probabilities']), np.array(inf['cuda_probabilities'])
    assert cpu.shape == gpu.shape and cpu.shape[0] == 64
    assert np.isfinite(cpu).all() and np.isfinite(gpu).all()
    assert np.allclose(cpu.sum(1), 1, atol=1e-6) and np.allclose(gpu.sum(1), 1, atol=1e-6)
    difference = float(np.abs(cpu-gpu).max())
    assert difference == inf['maximum_probability_difference'] <= 2e-4
    assert int((cpu.argmax(1) == gpu.argmax(1)).sum()) == inf['top1_agreement'] == 64
    assert len(reference['training']['losses']) == len(resumed['training']['losses']) == 32
    assert reference['training']['updates'] == 32 and resumed['training']['updates'] == 16
    assert max(resumed['resume']['maximum_differences'].values()) <= 1e-6
    assert np.allclose(reference['training']['losses'], resumed['training']['losses'], atol=1e-6, rtol=0)
    assert reference['checkpoint_files']['midpoint.pt'] == resumed['checkpoint_files']['midpoint.pt']
    for phase in (reference, resumed):
        assert all(v > 0 for v in phase['training']['first_step_gradient_l2'].values())
    if args.recompute_local or args.checkpoint_dir:
        import torch
        from mmso.cloud_pilot import CHECKPOINT, configure, new_model, tensor_tree_difference
        from mmso.joint_model import Vocabulary
        from mmso.joint_training import PairedCache
        configure(torch.device('cpu'))
    if args.recompute_local:
        import platform
        for relative, expected in bundle['files'].items():
            # Only the files that determine model/data semantics must match
            # locally; later reporting/runner edits do not change inference.
            if relative.startswith(('data/', 'mmso/')) or relative in {CHECKPOINT, str(Path(CHECKPOINT).with_name('config.json'))}:
                assert sha256(ROOT / relative) == expected, relative
        config = json.loads((ROOT / CHECKPOINT).with_name('config.json').read_text())
        model = new_model(config, torch.device('cpu')).eval()
        data = PairedCache(scenes, Vocabulary(config['vocabulary']), 'train')
        inputs, _ = data.batch(torch.arange(64), torch.device('cpu'))
        with torch.inference_mode():
            probabilities = (model(*inputs) / config['temperature']).softmax(-1).numpy()
        error = float(np.abs(probabilities-gpu).max())
        assert error <= 2e-4 and np.array_equal(probabilities.argmax(1), gpu.argmax(1))
        write_json(directory / 'local-parity.json', {'requests':64, 'top1_agreement':64,
                   'maximum_probability_difference':error, 'probabilities':probabilities.tolist(),
                   'torch':str(torch.__version__), 'platform':platform.platform(),
                   'checkpoint_sha256':sha256(ROOT / CHECKPOINT), 'bundle_sha256':sha256(directory / 'bundle.json')})
    if args.checkpoint_dir:
        checkpoints = {}
        for filename in ('midpoint.pt', 'reference.pt', 'resume.pt'):
            path = args.checkpoint_dir / filename
            hashes = {**reference['checkpoint_files'], **resumed['checkpoint_files']}
            assert sha256(path) == hashes[filename]
            checkpoints[filename] = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoints['midpoint.pt']['step'] == 16
        expected, actual = checkpoints['reference.pt'], checkpoints['resume.pt']
        assert expected['step'] == actual['step'] == 32
        differences = {key: tensor_tree_difference(expected[key], actual[key])
                       for key in ('model','optimizer','generator','cpu_rng','cuda_rng')}
        assert max(differences.values()) <= 1e-6
        assert expected['losses'] == reference['training']['losses']
        assert actual['losses'] == resumed['training']['losses']
        changed = tensor_tree_difference(checkpoints['midpoint.pt']['model'], actual['model'])
        assert changed > 0
        write_json(directory / 'downloaded-checkpoint-verification.json', {
            'hashes_verified': hashes, 'maximum_differences': differences,
            'model_change_after_midpoint': changed, 'step16_to_step32_verified': True})
    print(f'Cloud evidence verified: {checked} historical source/checkpoint files, 64 prediction vectors, two distinct invocations and 32-step resume.')


if __name__ == '__main__':
    main()
