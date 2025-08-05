import streamlit as st
import time
import subprocess
import requests

# Generate from OpenAI GPT-4.1
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants for Ollama
OLLAMA_MODEL = "deepseek-coder:1.3b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Function to check if Ollama is running
@st.cache_resource
def is_ollama_running():
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

# Generate from Deepseek using Ollama
def generate_from_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    start = time.time()
    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    elapsed_ms = (time.time() - start) * 1000
    if response.status_code == 200:
        return response.json().get("response", "No response returned."), elapsed_ms
    else:
        return f"Error: {response.text}", elapsed_ms



def generate_gpt4(prompt, max_tokens=512):
    start = time.time()
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    elapsed_ms = (time.time() - start) * 1000
    text = response.choices[0].message.content
    return text, elapsed_ms

# Streamlit UI
st.set_page_config(page_title="LLM Comparison", layout="wide")
st.title("🔍 Deepseek Coder (Ollama) vs OpenAI GPT-4.1")

if not is_ollama_running():
    st.error("❌ Ollama is not running. Please start it first using `ollama run deepseek-coder:1.3b`.")
    st.stop()

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Compare Responses"):
    with st.spinner("Deepseek Coder via Ollama..."):
        ds_response, ds_time = generate_from_ollama(prompt)
    with st.spinner("OpenAI GPT-4.1..."):
        gpt_response, gpt_time = generate_gpt4(prompt)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Deepseek Coder (Ollama)")
        st.code(ds_response)
        st.markdown(f"⏱️ **Response Time:** {ds_time:.2f} ms")

    with col2:
        st.subheader("🌐 OpenAI GPT-4.1")
        st.code(gpt_response)
        st.markdown(f"⏱️ **Response Time:** {gpt_time:.2f} ms")

st.markdown("---")
st.write("💡 Tip: Make sure your OpenAI key is set as the `OPENAI_API_KEY` environment variable.")
st.code("export OPENAI_API_KEY=your-key-here")
