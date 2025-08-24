export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

accelerate launch --config_file /examples/qwen_image/model_training/full/accelerate_config_fsdp-4gpu.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path ./pochacco \
  --dataset_metadata_path ./pochacco/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "/data/models/Qwen-Image/transformer/diffusion_pytorch_model*.safetensors,/data/models/Qwen-Image/text_encoder/model*.safetensors,/data/models/Qwen-Image/vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --gradient_accumulation_steps 4 \
  --use_8bit_paged_adam \
  --dataset_num_workers 4 \
  --prompt_cache_index /models/train/Qwen-Image_full/prompt_cache/index.json \
  --drop_text_encoder