# Uploading to Hugging Face Hub

## Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login   # paste a write-access token
```

## One-shot upload (recommended)

```bash
python -c "
from huggingface_hub import HfApi, create_repo

repo_id = 'YOUR_USERNAME/MERaLiON-2-3B-Medusa'   # change!

# Create the repo (public)
create_repo(repo_id, repo_type='model', exist_ok=True)

# Upload everything except this UPLOAD.md
api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id=repo_id,
    repo_type='model',
    commit_message='Initial Medusa adapter for MERaLiON-2-3B',
    ignore_patterns=['UPLOAD.md', '__pycache__/*', '*.pyc'],
)
print(f'done: https://huggingface.co/{repo_id}')
"
```

## Alternative: git LFS

```bash
cd /path/to/hf_medusa_pkg

git init
git lfs install
git lfs track "*.safetensors"
git add .gitattributes *.json *.md *.py meralion2_bl/*.py medusa_heads.safetensors

git remote add origin https://huggingface.co/YOUR_USERNAME/MERaLiON-2-3B-Medusa
git commit -m "Initial Medusa adapter for MERaLiON-2-3B"
git push -u origin main
```

## After upload

Verify loading from the Hub:

```python
import torch
from modeling_medusa import MERaLiON2MedusaForASR

model = MERaLiON2MedusaForASR.from_pretrained(
    "YOUR_USERNAME/MERaLiON-2-3B-Medusa",
    torch_dtype=torch.bfloat16,
).to("cuda")
```

Note: since the adapter references `MERaLiON/MERaLiON-2-3B` as its
base model, the **base weights (~7 GB) will be pulled from that repo on
first load** and cached under `~/.cache/huggingface/hub/`.

## Notes

- Total adapter size: ~42 MB (safetensors) + ~200 KB (bundled MERaLiON
  modeling code) + README + example.
- The bundled `meralion2_bl/` directory is a patched copy of MERaLiON-2's
  modeling code that works with current `transformers` versions (the
  upstream repo is pinned to an older transformers API and fails with
  `AttributeError: ... has no attribute '_supports_sdpa'` otherwise).
- No `trust_remote_code` is required: `modeling_medusa.py` imports the
  bundled code directly, side-stepping the upstream issue.
