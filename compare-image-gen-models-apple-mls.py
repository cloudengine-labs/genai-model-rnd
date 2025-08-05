import streamlit as st
# Replace diffusers import with MLX version
try:
    from diffusers import StableDiffusionPipeline
    import mlx.core as mx
    from mlx_stable_diffusion import StableDiffusionPipeline as MLXStableDiffusionPipeline
    MLX_AVAILABLE = True
except ImportError:
    from diffusers import StableDiffusionPipeline
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
        "mlx_id": "mlx-community/stable-diffusion-v1-5-4bit"  # MLX optimized version
    },
    "Stable Diffusion 2.1": {
        "local_path": "./models/sd21", 
        "hf_id": "stabilityai/stable-diffusion-2-1",
        "mlx_id": "mlx-community/stable-diffusion-2-1-4bit"
    },
    "Playground v2.5": {
        "local_path": "./models/playground",
        "hf_id": "playgroundai/playground-v2.5-1.0",
        "mlx_id": "mlx-community/playground-v2.5-1024px-aesthetic"
    }
}

def is_apple_silicon():
    return platform.system() == "Darwin" and platform.processor() == "arm"

@st.cache_resource(show_spinner=True)
def load_pipeline(config):
    model_path = config["local_path"]
    
    # Use MLX on Apple Silicon if available
    if is_apple_silicon() and MLX_AVAILABLE:
        try:
            # Try MLX optimized model first
            model_source = config.get("mlx_id", config["hf_id"])
            pipe = MLXStableDiffusionPipeline.from_pretrained(model_source)
            return pipe, "MLX"
        except Exception as e:
            st.warning(f"MLX loading failed, falling back to PyTorch: {e}")
    
    # Fallback to original PyTorch implementation
    model_source = model_path if os.path.exists(model_path) else config["hf_id"]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        return pipe, "CUDA"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        pipe = pipe.to("mps")  # Apple Metal Performance Shaders
        return pipe, "MPS"
    else:
        return pipe, "CPU"

# UI Setup
st.set_page_config(page_title="Prompt-to-Image Comparison", layout="wide")
st.title("🖼️ Compare Image Generation Across Models (Apple MLX + Local with Online Fallback)")

# Show system info
if is_apple_silicon():
    if MLX_AVAILABLE:
        st.success("🚀 Apple Silicon detected with MLX support - optimized performance enabled!")
    else:
        st.info("🍎 Apple Silicon detected. Install MLX for better performance: `pip install mlx-stable-diffusion`")
else:
    st.info("💻 Running on non-Apple hardware")

prompt = st.text_area("Enter your prompt:", "A futuristic cityscape at sunset")

selected_models = st.multiselect("Choose models to compare:", list(MODEL_CONFIG.keys()), default=list(MODEL_CONFIG.keys())[:2])

if st.button("Generate Images") and prompt:
    cols = st.columns(len(selected_models))

    for i, model_name in enumerate(selected_models):
        config = MODEL_CONFIG[model_name]
        with cols[i]:
            st.subheader(model_name)
            with st.spinner(f"Generating with {model_name}..."):
                pipe, backend = load_pipeline(config)
                start_time = time.time()
                
                # Generate image (MLX vs PyTorch have different APIs)
                if backend == "MLX":
                    image = pipe(prompt, num_inference_steps=20).images[0]
                else:
                    image = pipe(prompt).images[0]
                    
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
st.info("💾 This app uses MLX on Apple Silicon for optimal performance, with fallbacks to MPS/CUDA/CPU.")
st.write("To install MLX support:")
st.code("pip install mlx-stable-diffusion")
st.write("To preload models locally:")
st.code("pipe = StableDiffusionPipeline.from_pretrained('model_id')\npipe.save_pretrained('./models/model_name')")