# Building and uploading

This package is split into:

- **Code / docs** — committed in this repo subdir
- **Weights** — built from local checkpoints by `build.sh`

## 1. Build the distribution dir

```bash
GPTQ_MARLIN_DIR=quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin \
EAGLE_CKPT=eagle_best.pt \
BF16_BASE=/path/to/MERaLiON-2-3B \
OUT_DIR=hf_eagle_pkg/_built \
bash hf_eagle_pkg/build.sh
```

The script:
- Copies `README.md`, `requirements.txt`, `modeling_eagle.py`,
  `example_inference.py`, `eagle_model.py`, `meralion2_bl/`,
  `patch_autogptq_marlin_only.py`, `setup_cuda_includes.sh`.
- Converts `eagle_best.pt` → `eagle.safetensors` + `eagle_config.json`.
- Builds the standalone Gemma2 W4A16 dir under `text_decoder_w4a16/`
  if not cached.

## 2. Test locally

```bash
cd hf_eagle_pkg/_built
python example_inference.py /path/to/sample.wav --repo .
```

## 3. Upload

```bash
huggingface-cli login
huggingface-cli upload <user>/MERaLiON-2-3B-EAGLE-W4A16 \
    hf_eagle_pkg/_built .
```

For large files (model.safetensors), make sure `huggingface_hub` is
new enough to use the multi-part upload path, and that LFS is set up:

```bash
cd hf_eagle_pkg/_built
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
huggingface-cli repo create <user>/MERaLiON-2-3B-EAGLE-W4A16
git remote add origin https://huggingface.co/<user>/MERaLiON-2-3B-EAGLE-W4A16
git add .
git commit -m "Initial upload"
git push origin main
```

## Expected file sizes

| File | Size |
|---|---|
| `text_decoder_w4a16/model.safetensors` | ~1.3 GB (W4A16 of Gemma2-2.6B) |
| `base_bf16/model.safetensors`          | ~1.5 GB (Whisper encoder + adapter, BF16) |
| `eagle.safetensors`                    | ~85 MB  |
| code + configs                         | <1 MB   |
| **Total**                              | **~3 GB** |
