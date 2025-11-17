import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
import os
import sys

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.config import (
    model_config, app_config, eval_config, device, dtype
)
from src.utils.helpers import load_pipeline_for_evaluation

st.set_page_config(layout="wide")
st.title("Sketch to Face Translation System")

def get_pipeline(strategy_name, m_cfg, e_cfg, device, dtype):
    try:
        pipe = load_pipeline_for_evaluation(strategy_name, m_cfg, device, dtype)
        return pipe
    except FileNotFoundError as e:
        st.error(f"Cannot find model: {strategy_name}")
        return None
    except Exception as e:
        st.error(e)
        return None

def generate_image(pipe, condition_image, strategy_name, e_cfg):    
    is_adapter = (strategy_name == "strategy_4")
    
    pipe_args = {
        "prompt": e_cfg.prompt,
        "negative_prompt": e_cfg.negative_prompt,
        "image": condition_image,
        "num_inference_steps": e_cfg.num_inference_steps,
        "generator": torch.Generator(device=device).manual_seed(e_cfg.seed),
        "guidance_scale": e_cfg.guidance_scale
    }
    
    if is_adapter:
        pipe_args["adapter_conditioning_scale"] = e_cfg.adapter_conditioning_scale
    else:
        pipe_args["controlnet_conditioning_scale"] = e_cfg.controlnet_conditioning_scale
        
    return pipe(**pipe_args).images[0]


# Layout
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.header("Input Sketch")
    input_method = st.radio(
        "Choose input type:",
        ["Draw", "Upload"],
        horizontal=True
    )
    
    input_pil_image = None 

    if input_method == "Draw":
        canvas_result = st_canvas(
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            
            white_bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            white_bg.paste(pil_img, (0, 0), pil_img)
            input_pil_image = white_bg

    else: 
        uploaded_file = st.file_uploader("Choose sketch (PNG, JPG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            input_pil_image = Image.open(uploaded_file)
    
    if input_pil_image:
        st.image(input_pil_image, caption="Input sketch (512x512 resized )", width=512)
        input_pil_image = input_pil_image.resize((512, 512))

with col2:
    st.header("Output")
    
    if st.button("GENERATE", disabled=(input_pil_image is None), use_container_width=True, type="primary"):
        m_cfg = model_config()
        e_cfg = eval_config()
        
        strategies_to_run = {
            "Full-FT": "strategy_1",
            "LoRA-CN": "strategy_2",
            "LoRA-UNet": "strategy_3",
            "Adapter": "strategy_4",
        }
        
        output_images = {}
        
        with st.spinner("Generating image"):
            status_placeholder = st.empty()
            
            for i, (display_name, strategy_name) in enumerate(strategies_to_run.items()):
                
                pipe = get_pipeline(strategy_name, m_cfg, e_cfg, device, dtype)
                
                if pipe:
                    is_adapter = (strategy_name == "strategy_4")
                    condition_mode = "L" if is_adapter else "RGB"
                    condition_image = input_pil_image.convert(condition_mode)
                    
                    generated_image = generate_image(pipe, condition_image, strategy_name, e_cfg)
                    output_images[display_name] = generated_image
                    
                    del pipe
                    if device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    output_images[display_name] = None 
            
            status_placeholder.success("Completed!")
        
        out_col1, out_col2 = st.columns(2)
        out_col3, out_col4 = st.columns(2)
        
        cols = [out_col1, out_col2, out_col3, out_col4]
        
        for col, display_name in zip(cols, strategies_to_run.keys()):
            with col:
                st.subheader(display_name)
                img = output_images.get(display_name)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.error("Error")
    else:
        st.info("Please draw or upload an image and then choose 'Generate' button.")