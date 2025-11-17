import torch
import os
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel,
    StableDiffusionAdapterPipeline, T2IAdapter,
    UniPCMultistepScheduler
)
from peft import PeftModel

def load_pipeline_for_evaluation(strategy_name, m_cfg, device, dtype):
    
    strategy_cfg = getattr(m_cfg, strategy_name)
    model_path = os.path.join(strategy_cfg["output_dir"], "best_model")

    pipe = None
    
    if strategy_name == "strategy_1":
        controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            m_cfg.stable_diff_name, 
            controlnet=controlnet, 
            torch_dtype=dtype
        )

    elif strategy_name == "strategy_2":
        controlnet_base = ControlNetModel.from_pretrained(m_cfg.base_controlnet_name, torch_dtype=dtype)
        controlnet_peft = PeftModel.from_pretrained(controlnet_base, model_path)
        controlnet_merged = controlnet_peft.merge_and_unload()
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            m_cfg.stable_diff_name,
            controlnet=controlnet_merged,
            torch_dtype=dtype
        )

    elif strategy_name == "strategy_3":
        finetuned_controlnet_path = m_cfg.strategy_3["finetuned_controlnet_path"]            
        controlnet = ControlNetModel.from_pretrained(finetuned_controlnet_path, torch_dtype=dtype)
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            m_cfg.stable_diff_name,
            controlnet=controlnet,
            torch_dtype=dtype
        )
        
        pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path)

    elif strategy_name == "strategy_4":
        adapter = T2IAdapter.from_pretrained(model_path, torch_dtype=dtype)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            m_cfg.stable_diff_name,
            adapter=adapter,
            dtype=dtype
        )
    
    if pipe is None:
        raise ValueError(strategy_name)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe