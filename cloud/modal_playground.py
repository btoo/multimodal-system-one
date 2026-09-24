"""Scale-to-zero CPU serving for the protected MiSO playground."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('miso-playground')
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install('torch==2.14.0', 'numpy==2.5.3', 'Pillow==12.3.0',
                         'scipy==1.18.1', 'safetensors==0.8.0', 'fastapi==0.141.1', 'pydantic==2.13.5')
         .env({'PYTHONPATH':'/workspace:/workspace/cloud'})
         .add_local_dir(ROOT/'mmso', '/workspace/mmso', ignore=['__pycache__','*.pyc'])
         .add_local_file(Path(__file__), '/workspace/cloud/modal_playground.py')
         .add_local_file(ROOT/'artifacts/joint-full-v2/model.safetensors', '/workspace/artifacts/joint-full-v2/model.safetensors')
         .add_local_file(ROOT/'artifacts/joint-full-v2/config.json', '/workspace/artifacts/joint-full-v2/config.json')
         .add_local_file(ROOT/'artifacts/optimization-primitive-s24-v1/model.safetensors', '/workspace/artifacts/optimization-primitive-s24-v1/model.safetensors')
         .add_local_file(ROOT/'artifacts/optimization-primitive-s24-v1/config.json', '/workspace/artifacts/optimization-primitive-s24-v1/config.json'))

@app.function(image=image, secrets=[modal.Secret.from_name('miso-playground-auth')],
              cpu=(1,2), memory=(1024,2048), max_containers=1, min_containers=0,
              scaledown_window=20, timeout=90, startup_timeout=180, include_source=False)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(label='miso-native')
def serve():
    import os
    import torch
    from mmso.api.app import create_app
    if not os.environ.get('MMSO_API_KEY'):
        raise RuntimeError('Authenticated serving requires MMSO_API_KEY')
    torch.set_num_threads(2)
    return create_app(device='cpu')
