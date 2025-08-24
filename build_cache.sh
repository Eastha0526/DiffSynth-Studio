python build_prompt_cache.py \
  --metadata ./pochacco/metadata.csv \
  --index-path ./models/train/Qwen-Image_full/prompt_cache/index.json \
  --model-paths "/workspace/DiffSynth-Studio/models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model*.safetensors,/workspace/DiffSynth-Studio/models/Qwen/Qwen-Image/text_encoder/model*.safetensors,/workspace/DiffSynth-Studio/models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors" \
  --text-column caption \
  --device cpu \
  --batch-size 32
