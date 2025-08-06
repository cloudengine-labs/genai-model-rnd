import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import time
from PIL import Image
import os

# Use models from local folders (pre-downloaded) or fallback to online
MODEL_CONFIG = {
    "Stable Diffusion 1.5": {
        "local_path": "./models/sd15",
        "hf_id": "runwayml/stable-diffusion-v1-5"
    },
    "Stable Diffusion 2.1": {
        "local_path": "./models/sd21",
        "hf_id": "stabilityai/stable-diffusion-2-1"
    },
    # "Playground v2.5": {
    #     "local_path": "./models/playground",
    #     "hf_id": "playgroundai/playground-v2.5-1.0"
    # }
}

@st.cache_resource(show_spinner=True)
def load_pipeline(config):
    model_path = config["local_path"]
    model_source = model_path if os.path.exists(model_path) else config["hf_id"]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

# UI Setup
st.set_page_config(page_title="Prompt-to-Image Comparison", layout="wide")
st.title("🖼️ Compare Image Generation Across Models (Local with Online Fallback)")

prompt = st.text_area("Enter your prompt:", "A futuristic cityscape at sunset")

selected_models = st.multiselect("Choose models to compare:", list(MODEL_CONFIG.keys()), default=list(MODEL_CONFIG.keys())[:2])

if st.button("Generate Images") and prompt:
    cols = st.columns(len(selected_models))

    for i, model_name in enumerate(selected_models):
        config = MODEL_CONFIG[model_name]
        model_path = config["local_path"]
        with cols[i]:
            st.subheader(model_name)
            with st.spinner(f"Generating with {model_name}..."):
                pipe = load_pipeline(config)
                start_time = time.time()
                image = pipe(prompt).images[0]
                end_time = time.time()
                st.image(image, caption=f"⏱️ {end_time - start_time:.2f}s")
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
st.info("💾 This app uses local models if available, and falls back to Hugging Face if not.")
st.write("To preload models locally:")
st.code("pipe = StableDiffusionPipeline.from_pretrained('model_id')\npipe.save_pretrained('./models/model_name')")
