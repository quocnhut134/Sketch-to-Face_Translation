import torch
import lpips
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch_fidelity import calculate_metrics
from diffusers.utils import load_image

def calculate_lpips(real_dir, generated_dir, eval_cfg, device):
    loss_fn_vgg = lpips.LPIPS(net=eval_cfg.lpips_net).to(device)
    
    lpips_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    total_lpips_distance = 0
    image_count = 0
    
    generated_files = sorted(os.listdir(generated_dir))
    
    for i, filename in enumerate(tqdm(generated_files, desc="Calculating LPIPS")):
        if eval_cfg.max_samples and i >= eval_cfg.max_samples:
            break
            
        real_path = os.path.join(real_dir, filename)
        gen_path = os.path.join(generated_dir, filename)
        
        if not os.path.exists(real_path):
            print(f"Cannot find: {real_path}")
            continue
            
        real_img_pil = load_image(real_path)
        gen_img_pil = load_image(gen_path)
        
        real_tensor = lpips_transform(real_img_pil).to(device)
        gen_tensor = lpips_transform(gen_img_pil).to(device)
        
        with torch.no_grad():
            dist = loss_fn_vgg(real_tensor.unsqueeze(0), gen_tensor.unsqueeze(0))
        
        total_lpips_distance += dist.item()
        image_count += 1
            
    if image_count == 0:
        return 0.0, 0
        
    avg_lpips = total_lpips_distance / image_count
    return avg_lpips, image_count

def calculate_fid_kid(real_dir, generated_dir, image_count, eval_cfg):
    if image_count == 0:
        return {"frechet_inception_distance": 0, "kernel_inception_distance_mean": 0}
            
    metrics = calculate_metrics(
        input1=real_dir,
        input2=generated_dir,
        cuda=(torch.cuda.is_available()),
        fid=True,
        kid=True,
        input1_max_samples=image_count,
        input2_max_samples=image_count,
        kid_subset_size=min(image_count, 1000)
    )
    
    return metrics