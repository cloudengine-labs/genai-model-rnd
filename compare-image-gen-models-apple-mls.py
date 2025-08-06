import streamlit as st
# Replace diffusers import with MLX version
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import mlx.core as mx
    from mlx_image.stable_diffusion import StableDiffusionPipeline as MLXStableDiffusionPipeline
    MLX_AVAILABLE = True
except ImportError:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    MLX_AVAILABLE = False
    
import torch
import time
from PIL import Image
import os
import platform

# Use models from local folders (pre-downloaded) or fallback to online
MODEL_CONFIG = {
    "Stable Diffusion 1.5": {
        "local_path": "./models/sd15",
        "hf_id": "runwayml/stable-diffusion-v1-5",
        "mlx_id": "mlx-community/stable-diffusion-v1-5-4bit",
        "pipeline_class": StableDiffusionPipeline,
        "is_xl": False
    },
    "Stable Diffusion 2.1": {
        "local_path": "./models/sd21", 
        "hf_id": "stabilityai/stable-diffusion-2-1",
        "mlx_id": "mlx-community/stable-diffusion-2-1-4bit",
        "pipeline_class": StableDiffusionPipeline,
        "is_xl": False
    },
    # "Playground v2.5": {
    #     "local_path": "./models/playground",
    #     "hf_id": "playgroundai/playground-v2.5-1024px-aesthetic",
    #     "mlx_id": "mlx-community/playground-v2.5-1024px-aesthetic",
    #     "pipeline_class": StableDiffusionXLPipeline,
    #     "is_xl": True
    # }
}

def is_apple_silicon():
    return platform.system() == "Darwin" and platform.processor() == "arm"

@st.cache_resource(show_spinner=True)
def load_pipeline(config):
    model_path = config["local_path"]
    pipeline_class = config.get("pipeline_class", StableDiffusionPipeline)
    is_xl = config.get("is_xl", False)
    
    # Use MLX on Apple Silicon if available
    if is_apple_silicon() and MLX_AVAILABLE:
        try:
            # Try MLX optimized model first
            model_source = config.get("mlx_id", config["hf_id"])
            pipe = MLXStableDiffusionPipeline.from_pretrained(model_source)
            return pipe, "MLX", is_xl
        except Exception as e:
            st.warning(f"MLX loading failed, falling back to PyTorch: {e}")
    
    # Fallback to original PyTorch implementation
    model_source = model_path if os.path.exists(model_path) else config["hf_id"]
    
    # Load with appropriate pipeline class
    load_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
    }
    
    # Add SDXL-specific parameters
    if is_xl:
        load_kwargs.update({
            "use_safetensors": True,
            "variant": "fp16" if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else None
        })
    
    pipe = pipeline_class.from_pretrained(model_source, **load_kwargs)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        return pipe, "CUDA", is_xl
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        pipe = pipe.to("mps")  # Apple Metal Performance Shaders
        return pipe, "MPS", is_xl
    else:
        return pipe, "CPU", is_xl

def generate_image(pipe, prompt, backend, is_xl):
    """Generate image with appropriate parameters based on model type"""
    if backend == "MLX":
        # MLX parameters
        return pipe(prompt, num_inference_steps=20).images[0]
    elif is_xl:
        # SDXL parameters - typically works better with these settings
        return pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]
    else:
        # Standard SD parameters
        return pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

# UI Setup
st.set_page_config(page_title="Prompt-to-Image Comparison", layout="wide")
st.title("🖼️ Compare Image Generation Across Models (Apple MLX + SDXL Support)")

# Show system info
if is_apple_silicon():
    if MLX_AVAILABLE:
        st.success("🚀 Apple Silicon detected with MLX support - optimized performance enabled!")
    else:
        st.info("🍎 Apple Silicon detected. Install MLX for better performance: `pip install mlx mlx-image`")
else:
    st.info("💻 Running on non-Apple hardware")

# Model info display
with st.expander("📋 Model Information"):
    for name, config in MODEL_CONFIG.items():
        pipeline_type = "SDXL" if config["is_xl"] else "SD"
        resolution = "1024x1024" if config["is_xl"] else "512x512"
        st.write(f"**{name}**: {pipeline_type} | {resolution} | {config['pipeline_class'].__name__}")

prompt = st.text_area("Enter your prompt:", "A futuristic cityscape at sunset")

selected_models = st.multiselect(
    "Choose models to compare:", 
    list(MODEL_CONFIG.keys()), 
    default=list(MODEL_CONFIG.keys())[:2]
)

if st.button("Generate Images") and prompt:
    cols = st.columns(len(selected_models))

    for i, model_name in enumerate(selected_models):
        config = MODEL_CONFIG[model_name]
        with cols[i]:
            st.subheader(model_name)
            pipeline_type = "SDXL" if config["is_xl"] else "SD"
            resolution = "1024x1024" if config["is_xl"] else "512x512"
            st.caption(f"{pipeline_type} | {resolution}")
            
            with st.spinner(f"Generating with {model_name}..."):
                pipe, backend, is_xl = load_pipeline(config)
                start_time = time.time()
                
                # Generate image with model-appropriate parameters
                image = generate_image(pipe, prompt, backend, is_xl)
                    
                end_time = time.time()
                
                st.image(image, caption=f"⏱️ {end_time - start_time:.2f}s ({backend})")
                
                # Save image to buffer for download
                image_path = f"generated_{model_name.replace(' ', '_')}.png"
                image.save(image_path)
                with open(image_path, "rb") as f:
                    st.download_button(
                        label="Download Image",
                        data=f,
                        file_name=image_path,
                        mime="image/png"
                    )

st.markdown("---")
st.info("💾 This app supports both SD and SDXL models with MLX optimization on Apple Silicon.")

col1, col2 = st.columns(2)
with col1:
    st.write("**To install MLX support:**")
    st.code("pip install mlx mlx-image")

with col2:
    st.write("**To preload models locally:**")
    st.code("""# For SD models
pipe = StableDiffusionPipeline.from_pretrained('model_id')
pipe.save_pretrained('./models/model_name')

# For SDXL models  
pipe = StableDiffusionXLPipeline.from_pretrained('model_id')
pipe.save_pretrained('./models/model_name')""")