#!/usr/bin/env python3

"""
Debug script to identify the model loading issue
"""

import json
import os
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def debug_model_loading():
    """Debug the model loading process step by step"""
    
    # Test the model configuration parsing
    model_id_with_origin_paths = "Qwen/Qwen-Image:transformer/diffusion_pytorch_model.safetensors,Qwen/Qwen-Image:text_encoder/model.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
    
    print(f"Original input: {model_id_with_origin_paths}")
    
    # Parse the paths
    model_id_with_origin_paths_list = model_id_with_origin_paths.split(",")
    print(f"Split paths: {model_id_with_origin_paths_list}")
    
    # Create model configs
    model_configs = []
    for path in model_id_with_origin_paths_list:
        config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern=path.strip())
        model_configs.append(config)
        print(f"Created ModelConfig: model_id='Qwen/Qwen-Image', origin_file_pattern='{path.strip()}'")
    
    print(f"Total model configs: {len(model_configs)}")
    
    # Try to create the pipeline and catch the exact error
    try:
        print("\nAttempting to create QwenImagePipeline...")
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype="bfloat16", 
            device="cpu", 
            model_configs=model_configs
        )
        print("SUCCESS: Pipeline created successfully!")
        
    except Exception as e:
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")
        
        # Print detailed traceback
        import traceback
        print("\nDETAILED TRACEBACK:")
        traceback.print_exc()
        
        # Try to identify where the list is being created
        print("\nDEBUGGING MODEL CONFIGS:")
        for i, config in enumerate(model_configs):
            print(f"Config {i}:")
            for attr in dir(config):
                if not attr.startswith('_'):
                    try:
                        value = getattr(config, attr)
                        print(f"  {attr}: {type(value)} = {value}")
                    except:
                        print(f"  {attr}: <unable to access>")

def test_alternative_approach():
    """Test using exact model paths instead"""
    print("\n" + "="*50)
    print("TESTING ALTERNATIVE APPROACH WITH EXACT PATHS")
    print("="*50)
    
    try:
        # Try with a single model config first
        model_configs = [ModelConfig(model_id="Qwen/Qwen-Image")]
        
        print("Attempting with single ModelConfig (no origin_file_pattern)...")
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype="bfloat16",
            device="cpu", 
            model_configs=model_configs
        )
        print("SUCCESS: Basic pipeline creation worked!")
        
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    print("DEBUGGING QWEN-IMAGE MODEL LOADING")
    print("="*50)
    
    debug_model_loading()
    test_alternative_approach()
    
    print("\nIf this script reveals the issue, try one of these solutions:")
    print("1. Use the fixed training script with exact file names")
    print("2. Download model files locally and use --model_paths parameter")
    print("3. Update DiffSynth-Studio to latest version")
    print("4. Check if the model files exist in the expected locations")