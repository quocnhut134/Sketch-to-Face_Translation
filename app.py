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
    model_config, eval_config, device, dtype
)
from src.utils.helpers import load_pipeline_for_evaluation

st.set_page_config(layout="wide")
st.title("Sketch to Face Translation System")

with st.sidebar:
    st.header("Configuration")
    
    # Prompts
    prompt = st.text_area(
        "Positive Prompt",
        eval_config.prompt,
        height=100
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        eval_config.negative_prompt, 
        height=150
    )
    
    # Sliders
    guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, eval_config.guidance_scale, 0.5)
    control_scale = st.slider("ControlNet Scale", 0.0, 1.0, eval_config.controlnet_conditioning_scale, 0.1)
    adapter_scale = st.slider("Adapter Scale", 0.0, 1.0, eval_config.adapter_conditioning_scale, 0.1)
    
    run_button = st.button("GENERATE", use_container_width=True, type="primary")


@st.cache_data(show_spinner=False) 
def get_pipeline(_strategy_name, _m_cfg, _device, _dtype):
    try:
        pipe = load_pipeline_for_evaluation(_strategy_name, _m_cfg, _device, _dtype)
        return pipe
    except FileNotFoundError as e:
        st.error(_strategy_name)
        return None
    except Exception as e:
        st.error(e)
        return None

def generate_image(pipe, condition_image, strategy_name, e_cfg, custom_control_scale, custom_adapter_scale):    
    is_adapter = (strategy_name == "strategy_4")
    
    pipe_args = {
        "prompt": prompt, 
        "negative_prompt": negative_prompt, 
        "image": condition_image,
        "num_inference_steps": e_cfg.num_inference_steps,
        "generator": torch.Generator(device=device).manual_seed(e_cfg.seed),
        "guidance_scale": guidance_scale 
    }
    
    if is_adapter:
        pipe_args["adapter_conditioning_scale"] = custom_adapter_scale 
    else:
        pipe_args["controlnet_conditioning_scale"] = custom_control_scale 
        
    return pipe(**pipe_args).images[0]

# Layout
col1, col2 = st.columns([0.4, 0.6]) 

with col1:
    st.header("1. Input Sketch")
    input_method = st.radio(
        "Choose input type:",
        ["Draw", "Upload"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    input_pil_image = None 

    if input_method == "Draw":
        st.write("Draw sketch:")
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
            input_pil_image = white_bg.resize((512, 512))

    else: 
        uploaded_file = st.file_uploader("Choose sketch file (PNG, JPG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            input_pil_image = Image.open(uploaded_file).resize((512, 512))
    
    st.markdown("---")
    st.markdown("**Input:**")
    input_image_placeholder = st.empty()
    if input_pil_image:
        input_image_placeholder.image(input_pil_image, caption="Input sketch (resized to 512x512)", use_container_width=True)
    else:
        input_image_placeholder.info("Please draw or upload sketch")

with col2:
    st.header("2. Generated Images")
    
    out_col1, out_col2 = st.columns(2)
    out_col3, out_col4 = st.columns(2)
    
    placeholder_s1 = out_col1.empty()
    placeholder_s2 = out_col2.empty()
    placeholder_s3 = out_col3.empty()
    placeholder_s4 = out_col4.empty()
    
    placeholder_s1.info("Full Finetuning ControlNet")
    placeholder_s2.info("LoRA on ControlNet")
    placeholder_s3.info("LoRA on UNet combine finetuned full ControlNet")
    placeholder_s4.info("T2I Adapter")

if run_button and input_pil_image:
    m_cfg = model_config()
    e_cfg = eval_config()
    
    strategies_to_run = {
        "Strategy 1 (Full-FT)": {
            "name": "strategy_1",
            "placeholder": placeholder_s1,
            "is_adapter": False
        },
        "Strategy 2 (LoRA-CN)": {
            "name": "strategy_2",
            "placeholder": placeholder_s2,
            "is_adapter": False
        },
        "Strategy 3 (LoRA-UNet)": {
            "name": "strategy_3",
            "placeholder": placeholder_s3,
            "is_adapter": False
        },
        "Strategy 4 (Adapter)": {
            "name": "strategy_4",
            "placeholder": placeholder_s4,
            "is_adapter": True
        }
    }
    
    for display_name, config in strategies_to_run.items():
        pipe = get_pipeline(config["name"], m_cfg, device, dtype)
        
        if pipe:
            with st.spinner(f"Generating image with {display_name}..."):
                condition_mode = "L" if config["is_adapter"] else "RGB"
                condition_image = input_pil_image.convert(condition_mode)
                
                generated_image = generate_image(
                    pipe=pipe,
                    condition_image=condition_image,
                    strategy_name=config["name"],
                    e_cfg=e_cfg,
                    custom_control_scale=control_scale, 
                    custom_adapter_scale=adapter_scale 
                )
                
                config["placeholder"].image(generated_image, caption=display_name, use_container_width=True)
                
        else:
            config["placeholder"].error(f"Error with {display_name}.")
    
    st.success("Completed!")