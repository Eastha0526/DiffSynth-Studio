import os, json, hashlib, argparse, torch
from safetensors.torch import save_file
from tqdm import tqdm
import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ImageDataset, ModelLogger, launch_training_task, qwen_image_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern=i.strip()) for i in model_id_with_origin_paths]
        if tokenizer_path is not None:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=ModelConfig(tokenizer_path))
        else:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        self.pipe.text_encoder.to("cuda:0")
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        
        # CFG-unsensitive parameters
        inputs_shared = {}
        inputs_nega = {}
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:# inputs_posi / inputs_nega -> {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}, each is tensor on device.
            # inputs_shared -> not to be cached.
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

def _slugify(prompt: str) -> str:
    # Stable, short, filesystem-safe filename
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]

@torch.no_grad()
def build_prompt_cache(
    args,
    out_dir: str,
    index_json: str,
    dtype_out: torch.dtype = torch.bfloat16,
):
    os.makedirs(out_dir, exist_ok=True)

    # Build a tiny model instance on CPU to run preprocessing/encoding
    dataset = ImageDataset(args=args)
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
    )
    model.eval()

    index = {}
    seen_prompts = set()

    for i in tqdm(range(len(dataset)), desc="Caching prompts"):
        ex = dataset[i]
        prompt = ex["prompt"]
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        out = model.forward_preprocess(ex)

        if "prompt_emb" not in out or "prompt_emb_mask" not in out:
            raise RuntimeError(
                "forward_preprocess did not produce 'prompt_emb'/'prompt_emb_mask'. "
                "Make sure your units produce these keys."
            )

        prompt_emb = out["prompt_emb"].detach().to(dtype_out).cpu()
        prompt_emb_mask = out["prompt_emb_mask"].detach().to(torch.int64).cpu()

        fn = f"{_slugify(prompt)}.safetensors"
        fpath = os.path.join(out_dir, fn)

        save_file(
            {"prompt_emb": prompt_emb, "prompt_emb_mask": prompt_emb_mask},
            fpath,
            metadata={
                "prompt": prompt,
                "dtype": str(dtype_out),
                "emb_shape": str(list(prompt_emb.shape)),
                "mask_shape": str(list(prompt_emb_mask.shape)),
                "tokenizer_path": str(args.tokenizer_path),
            },
        )
        index[prompt] = fpath

    with open(index_json, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(index)} prompt entries to {index_json}")

if __name__ == "__main__":
    p = qwen_image_parser()
    p.add_argument("--prompt_cache_dir", type=str, required=True)
    p.add_argument("--prompt_cache_index", type=str, required=True)
    p.add_argument("--cache_dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.cache_dtype]
    build_prompt_cache(args, args.prompt_cache_dir, args.prompt_cache_index, dtype_out=dtype)
    # 1MB per cache.