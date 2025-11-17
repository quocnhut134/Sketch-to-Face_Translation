import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
from diffusers.utils import load_image
import json

from configs.config import (
    data_config, model_config, eval_config, 
    device, dtype
)
from src.utils.helpers import load_pipeline_for_evaluation
from src.evaluation.metrics import calculate_lpips, calculate_fid_kid

def generate_images(pipe, strategy_name, d_cfg, e_cfg):    
    sketch_dir = os.path.join(d_cfg.dataset_dir, "test", "sketches")
    generated_dir = e_cfg.generated_dir
    os.makedirs(generated_dir, exist_ok=True)
    
    print(f"Input: {sketch_dir}")
    print(f"Output: {generated_dir}")

    generator = torch.Generator(device=device).manual_seed(e_cfg.seed)
    sketch_files = sorted(os.listdir(sketch_dir))
    
    is_adapter = (strategy_name == "strategy_4")
    condition_mode = "L" if is_adapter else "RGB"
    
    for i, filename in enumerate(tqdm(sketch_files, desc="Generating Images")):
        if e_cfg.max_samples and i >= e_cfg.max_samples:
            break
            
        sketch_path = os.path.join(sketch_dir, filename)
        generated_path = os.path.join(generated_dir, filename)
        
        condition_image = load_image(sketch_path).convert(condition_mode).resize((512, 512))
        
        pipe_args = {
            "prompt": e_cfg.prompt,
            "negative_prompt": e_cfg.negative_prompt,
            "image": condition_image,
            "num_inference_steps": e_cfg.num_inference_steps,
            "generator": generator,
            "guidance_scale": e_cfg.guidance_scale
        }
        
        if is_adapter:
            pipe_args["adapter_conditioning_scale"] = e_cfg.adapter_conditioning_scale
        else:
            pipe_args["controlnet_conditioning_scale"] = e_cfg.controlnet_conditioning_scale
            
        generated_image_pil = pipe(**pipe_args).images[0]
        generated_image_pil.save(generated_path)

def main(strategy_name):
    print(strategy_name)
    
    d_cfg = data_config()
    m_cfg = model_config()
    e_cfg = eval_config()
    
    pipe = load_pipeline_for_evaluation(strategy_name, m_cfg, device, dtype)

    generate_images(pipe, strategy_name, d_cfg, e_cfg)

    real_dir = os.path.join(d_cfg.dataset_dir, "test", "photos")
    generated_dir = e_cfg.generated_dir
    
    # LPIPS
    avg_lpips, image_count = calculate_lpips(real_dir, generated_dir, e_cfg, device)
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    # FID/KID
    metrics = calculate_fid_kid(real_dir, generated_dir, image_count, e_cfg)
    fid = metrics.get('frechet_inception_distance', 0)
    kid_mean = metrics.get('kernel_inception_distance_mean', 0)
    
    print(f"FID: {fid:.4f}")
    print(f"KID Mean: {kid_mean:.4f}")

    results = {
        "strategy": strategy_name,
        "model_path": os.path.join(getattr(m_cfg, strategy_name)["output_dir"], "best_model"),
        "image_count": image_count,
        "avg_lpips": avg_lpips,
        "fid": fid,
        "kid_mean": kid_mean
    }
    
    results_path = os.path.join(e_cfg.generated_dir, f"metrics_{strategy_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sketch-to-face model")
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True, 
        choices=["strategy_1", "strategy_2", "strategy_3", "strategy_4"],
        help="Choose trained strategy to evaluate"
    )
    args = parser.parse_args()
    
    main(args.strategy)