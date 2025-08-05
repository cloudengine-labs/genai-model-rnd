import streamlit as st
import openai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set your OpenAI API key securely (recommended: use Streamlit secrets or env vars)
openai.api_key = ""

@st.cache_resource
def load_deepseek_model():
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer_ds, model_ds = load_deepseek_model()

st.title("🤖 Prompt Comparison: Deepseek Coder vs GPT-4.1")
st.markdown("Enter a prompt to compare results from a local coder model and GPT-4.1.")

prompt = st.text_area("📝 Prompt", height=150)

if st.button("Compare Responses") and prompt:
    # Deepseek Coder response
    with st.spinner("Running Deepseek Coder..."):
        start_ds = time.time()
        inputs = tokenizer_ds(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        outputs = model_ds.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            pad_token_id=tokenizer_ds.eos_token_id
        )
        deepseek_text = tokenizer_ds.decode(outputs[0], skip_special_tokens=True)
        end_ds = time.time()
        time_ds = (end_ds - start_ds) * 1000

    # OpenAI GPT-4.1 response
    with st.spinner("Querying GPT-4.1..."):
        start_gpt = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
        )
        gpt_text = response['choices'][0]['message']['content']
        end_gpt = time.time()
        time_gpt = (end_gpt - start_gpt) * 1000

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Deepseek Coder")
        st.code(deepseek_text, language="yaml")
        st.caption(f"⏱ {time_ds:.2f} ms")

    with col2:
        st.subheader("💡 GPT-4.1")
        st.code(gpt_text, language="yaml")
        st.caption(f"⏱ {time_gpt:.2f} ms")
