import cv2
import numpy as np
import os
import glob
import random
import sys
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from configs.config import data_config

def augment_sketch(sketch_image, d_cfg):
    augmented_sketch = sketch_image.copy()

    if random.random() > d_cfg.augment_prob:
        return augmented_sketch

    # Erosion
    if random.random() < d_cfg.erosion_prob:
        kernel = np.ones((d_cfg.erosion_size, d_cfg.erosion_size), np.uint8)
        augmented_sketch = cv2.erode(augmented_sketch, kernel, iterations=1)
        
    # Dropout
    if random.random() < d_cfg.dropout_prob:
        rows, cols, _ = augmented_sketch.shape
        num_holes = random.randint(int(d_cfg.dropout_holes * 0.8), d_cfg.dropout_holes)
        
        for _ in range(num_holes):
            hole_w = random.randint(d_cfg.dropout_min_size, d_cfg.dropout_max_size) 
            hole_h = random.randint(d_cfg.dropout_min_size, d_cfg.dropout_max_size)
            
            x1 = random.randint(0, cols - hole_w)
            y1 = random.randint(0, rows - hole_h)
            
            fill = (d_cfg.dropout_fill_value, d_cfg.dropout_fill_value, d_cfg.dropout_fill_value)
            augmented_sketch[y1:y1+hole_h, x1:x1+hole_w] = fill
            
    return augmented_sketch

def apply_hed(image, net, target_size, d_cfg, apply_augmentation=False):
    (h, w) = image.shape[:2]
    mean_pixel_values = (104.00698793, 116.66876762, 122.67891434)
    blob = cv2.dnn.blobFromImage(image, 
                                 scalefactor=1.0, 
                                 size=(w, h), 
                                 mean=mean_pixel_values,
                                 swapRB=False, 
                                 crop=False)
    net.setInput(blob)
    hed_output = net.forward()
    hed_output = hed_output[0, 0] 
    hed_output = cv2.resize(hed_output, (w, h))
    hed_output = cv2.normalize(hed_output, None, 0, 255, cv2.NORM_MINMAX)
    hed_output = hed_output.astype("uint8")
    hed_output = 255 - hed_output
    hed_output_rgb = cv2.cvtColor(hed_output, cv2.COLOR_GRAY2BGR)
    
    if apply_augmentation:
        hed_output_rgb = augment_sketch(hed_output_rgb, d_cfg)
    return hed_output_rgb

def main():
    d_cfg = data_config()    
    net = cv2.dnn.readNetFromCaffe(d_cfg.hed_prototxt, d_cfg.hed_model)
    all_image_paths = sorted(glob.glob(f"{d_cfg.creation_ffhq_dir}/**/*.png", recursive=True))

    print(f"Total images: {len(all_image_paths)}")
    
    random.seed(42)
    random.shuffle(all_image_paths)

    num_total = len(all_image_paths)
    num_train = int(num_total * d_cfg.train_ratio)
    num_eval = int(num_total * d_cfg.eval_ratio)

    splits = {
        "train": all_image_paths[:num_train],
        "val": all_image_paths[num_train : num_train + num_eval],
        "test": all_image_paths[num_train + num_eval:],
    }

    for split_name, file_list in splits.items():    
        target_dir = os.path.join(d_cfg.creation_output_dir, split_name, "photos")
        source_dir = os.path.join(d_cfg.creation_output_dir, split_name, "sketches")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(source_dir, exist_ok=True)
        
        should_augment = (split_name == "train")
        
        for img_path in tqdm(file_list, desc=f"Processing {split_name}"):
            photo_original = cv2.imread(img_path)
            if photo_original is None:
                continue
            
            photo_resized = cv2.resize(photo_original, 
                                        (d_cfg.img_size, d_cfg.img_size), 
                                        interpolation=cv2.INTER_AREA)
            
            sketch_hed = apply_hed(photo_resized, 
                                    net, 
                                    target_size=d_cfg.img_size, 
                                    d_cfg=d_cfg, 
                                    apply_augmentation=should_augment)
            
            filename = os.path.basename(img_path)
            
            target_save_path = os.path.join(target_dir, filename)
            source_save_path = os.path.join(source_dir, filename)
            
            cv2.imwrite(target_save_path, photo_resized)
            cv2.imwrite(source_save_path, sketch_hed)

if __name__ == "__main__":
    main()