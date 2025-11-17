import torch
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data_dir")
saved_models_dir = os.path.join(project_root, "saved_models")
outputs_dir = os.path.join(project_root, "outputs")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


class data_config:
    # Create data
    creation_ffhq_dir = os.path.join(data_dir, "ffhq")
    creation_output_dir = os.path.join(data_dir, "large_hed-augmented_ffhq_dataset")
    hed_prototxt = os.path.join(saved_models_dir, "hed_model", "deploy.prototxt")
    hed_model = os.path.join(saved_models_dir, "hed_model", "hed_pretrained_bsds.caffemodel")
    train_ratio = 0.8  
    eval_ratio = 0.1  
    img_size = 512 

    # Data Augmentation
    augment_prob = 0.95
    erosion_prob = 1.0
    erosion_size = 6
    dropout_prob = 1.0
    dropout_holes = 65536
    dropout_min_size = 1
    dropout_max_size = 3     
    dropout_fill_value = 255 
    
    # Training
    dataset_dir = creation_output_dir

class model_config:
    stable_diff_name = "botp/stable-diffusion-v1-5"
    base_controlnet_name = "lllyasviel/sd-controlnet-hed"
    base_t2i_adapter_name = "TencentARC/t2iadapter_sketch_sd15v2"
    
    # Full ControlNet Finetune
    strategy_1 = {
        "learning_rate": 5e-6,
        "weight_decay": 1e-4,
        "output_dir": os.path.join(saved_models_dir, "strategy_1_full_controlnet")
    }

    # LoRA on ControlNet
    strategy_2 = {
        "learning_rate": 2e-4,
        "weight_decay": 1e-2,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2", "conv_in"],
        "output_dir": os.path.join(saved_models_dir, "strategy_2_lora_controlnet")
    }
    
    # Finetuned ControlNet + LoRA on UNet
    strategy_3 = {
        "finetuned_controlnet_path": os.path.join(strategy_1["output_dir"], "best_model"),
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2"],
        "output_dir": os.path.join(saved_models_dir, "strategy_3_lora_unet")
    }
    
    # T2I Adapter Finetune
    strategy_4 = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "output_dir": os.path.join(saved_models_dir, "strategy_4_t2i_adapter")
    }

class train_config:    
    device = device
    dtype = dtype
    num_epochs = 20
    batch_size = 4 
    accumulation_steps = 8
    patience = 5  
    
    # Prompts
    default_prompt = "a realistic photo of a human face"
    null_prompt_prob = 0.1 
    
    # Tokenizer
    padding = "max_length"
    return_tensors = "pt"

class eval_config:    
    generated_dir = os.path.join(outputs_dir, "generated_for_metrics")
    
    max_samples = 500
    seed = 1234
    lpips_net = 'vgg'

    # Prompts
    prompt = """(hyper-realistic photo:1.2), (ultra-detailed skin texture:1.1), 
                detailed pores, realistic eyes, sharp focus, 
                8k UHD, professional studio lighting, DSLR"""
    negative_prompt = """(drawing:1.4), (sketch:1.4), (painting:1.3), cartoon, 3D, 
                        render, CGI, anime, illustration, (deformed:1.2), (disfigured:1.2), 
                        ugly, bad anatomy, (blurry:1.1), low quality, low-res"""
    
    # Inference
    num_inference_steps = 30 
    guidance_scale = 7.5 
    controlnet_conditioning_scale = 0.9 
    adapter_conditioning_scale = 0.9

class app_config:    
    controlnet_path = model_config.strategy_1["output_dir"]
    base_model_name = model_config.stable_diff_name
    annotator_model_name = 'lllyasviel/Annotators'
    
    device = device
    dtype = dtype