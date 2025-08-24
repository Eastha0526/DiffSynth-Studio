python build_prompt_cache.py \
  --metadata ./pochacco/metadata.csv \
  --index-path /workspace/DiffSynth-Studio/models/train/Qwen-Image_full/prompt_cache/index.json \
  --text-encoder-path /workspace/DiffSynth-Studio/models/Qwen/Qwen-Image/text_encoder \
  --tokenizer-path    /workspace/DiffSynth-Studio/models/Qwen/Qwen-Image/tokenizer \
  --text-column prompt \
  --device cpu \
  --batch-size 32