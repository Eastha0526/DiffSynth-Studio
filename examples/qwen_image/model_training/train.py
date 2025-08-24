import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ImageDataset, ModelLogger, launch_training_task, qwen_image_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from safetensors.torch import load_file

def maybe_compile_dit(dit, enable: bool = True):
    if not enable:
        return dit
    dit.img_in   = torch.compile(dit.img_in,   dynamic=True, fullgraph=False, mode="max-autotune")
    dit.txt_in   = torch.compile(dit.txt_in,   dynamic=True, fullgraph=False, mode="max-autotune")
    dit.proj_out = torch.compile(dit.proj_out, dynamic=True, fullgraph=False, mode="max-autotune")
    dit.norm_out = torch.compile(dit.norm_out, dynamic=True, fullgraph=False, mode="max-autotune")

    for i, blk in enumerate(dit.transformer_blocks):
        dit.transformer_blocks[i] = torch.compile(blk, dynamic=True, fullgraph=False, mode="max-autotune")
    return dit



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
        prompt_cache_index: str = None,
        drop_text_encoder: bool = False,
        enable_fp8_attention: bool = False,
        compile: bool = True
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

        # Reset training scheduler (do it in each training step)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        self.enable_fp8_attention = enable_fp8_attention
        maybe_compile_dit(self.pipe.dit, False)
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
        self.prompt_cache_index = None
        if prompt_cache_index is not None and os.path.isfile(prompt_cache_index):
            with open(prompt_cache_index, "r", encoding="utf-8") as f:
                self.prompt_cache_index = json.load(f)
            print(f"[prompt-cache] Loaded {len(self.prompt_cache_index)} entries from {prompt_cache_index}")

        if self.prompt_cache_index is not None and drop_text_encoder:
            if hasattr(self.pipe, "in_iteration_models"):
                self.pipe.in_iteration_models = [n for n in self.pipe.in_iteration_models if n not in ("text_encoder", "tokenizer")]
            if hasattr(self.pipe, "text_encoder"):
                try:
                    for p in self.pipe.text_encoder.parameters(): p.requires_grad_(False)
                except Exception:
                    pass
                self.pipe.text_encoder = self.pipe.text_encoder.to('cpu')
                self.pipe.text_encoder = None
            if hasattr(self.pipe, "tokenizer"):
                self.pipe.tokenizer = None
            if hasattr(self.pipe, "units"):
                kept = []
                for u in self.pipe.units:
                    uname = getattr(u, "name", u.__class__.__name__).lower()
                    if ("text" in uname) or ("prompt" in uname and "encode" in uname) or ("token" in uname):
                        print(f"[prompt-cache] Removing {uname} from pipeline.")
                        continue
                    kept.append(u)
                self.pipe.units = kept
            print("[prompt-cache] Removed text encoder/tokenizer from pipeline.")

    def _load_cached_prompt(self, prompt: str):
        if self.prompt_cache_index is None:
            return None
        fpath = self.prompt_cache_index.get(prompt, None)
        if fpath is None or not os.path.isfile(fpath):
            raise KeyError(f"[prompt-cache] Missing cache entry for prompt: {prompt!r}")
        t = load_file(fpath, device="cpu") 
        pe = t["prompt_emb"]
        pm = t["prompt_emb_mask"].to(torch.int64) 
        return pe, pm

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "enable_fp8_attention": self.enable_fp8_attention
        }
        
        # Extra inputs
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        if len(blockwise_controlnet_input) > 0:
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]
        if self.prompt_cache_index is not None:
            pe, pm = self._load_cached_prompt(data["prompt"])
            inputs_posi = {"prompt_emb": pe, "prompt_emb_mask": pm}
            inputs_nega = {}  # ignored
        else:
            inputs_posi = {"prompt": data["prompt"]}
            inputs_nega = {"negative_prompt": ""}
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



if __name__ == "__main__":
    parser = qwen_image_parser()
    parser.add_argument("--prompt_cache_index", type=str, default=None)
    parser.add_argument("--drop_text_encoder", action="store_true")
    args = parser.parse_args()
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
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        drop_text_encoder=args.drop_text_encoder,
        prompt_cache_index=args.prompt_cache_index,
        enable_fp8_attention=False
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        print("Using AdamW8bit optimizer")
        optimizer = bnb.optim.AdamW8bit(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.use_8bit_paged_adam:
        import bitsandbytes as bnb
        print("Using PagedAdamW8bit optimizer")
        optimizer = bnb.optim.PagedAdamW8bit(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # print the current vram usage state
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
    )