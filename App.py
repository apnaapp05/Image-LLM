# app.py
"""
Streamlit app that:
- Generates text using an OpenAI LLM
- Generates images from prompts
Works both locally and on Streamlit Cloud.
"""

import streamlit as st
import os
import base64
import io
from PIL import Image
import requests

# --- Load API key (from Streamlit Secrets or environment variable) ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.set_page_config(page_title="Missing API Key", layout="centered")
    st.title("🚨 API Key Missing")
    st.error(
        "No OpenAI API key found.\n\n"
        "To fix this:\n"
        "- If you’re running locally: create a `.env` file with `OPENAI_API_KEY=sk-...`\n"
        "- If you’re on Streamlit Cloud: go to *Manage app → Settings → Secrets* "
        "and add your key like this:\n\n"
        "```\nOPENAI_API_KEY = \"sk-your-key\"\n```"
    )
    st.stop()

# Import openai after key is verified
import openai
openai.api_key = OPENAI_API_KEY


# --- Helper functions ---
def call_llm(prompt, system_prompt, model="gpt-4o"):
    """Generate text using the LLM."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=400,
            temperature=0.8,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error calling LLM: {e}"


def generate_images(prompt, model="gpt-image-1", size="1024x1024", n=1):
    """Generate image(s) from text prompt."""
    try:
        response = openai.Image.create(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            response_format="b64_json"
        )
        images = []
        for item in response["data"]:
            img_data = base64.b64decode(item["b64_json"])
            images.append(img_data)
        return images
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return []


# --- Streamlit UI ---
st.set_page_config(page_title="LLM + Image Generator", layout="wide")
st.title("🎨 LLM + Image Generator")

with st.sidebar:
    st.header("⚙️ Settings")
    llm_model = st.text_input("LLM Model", value="gpt-4o")
    image_model = st.text_input("Image Model", value="gpt-image-1")
    image_size = st.selectbox("Image Size", ["256x256", "512x512", "1024x1024"], index=2)
    system_prompt = st.text_area(
        "System Prompt (optional)",
        value="You are a helpful assistant that writes concise, creative captions and descriptions."
    )
    n_images = st.slider("Number of images", 1, 4, 1)

st.subheader("🧠 Text Generation")
user_text_prompt = st.text_area(
    "Enter a prompt for the LLM:",
    value="Write a short, vivid product description for a cozy smart lamp called 'Lumi'."
)
if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        text_output = call_llm(user_text_prompt, system_prompt, llm_model)
    st.success("Done!")
    st.write("### LLM Output:")
    st.write(text_output)
    st.download_button("Download Text", text_output, file_name="llm_output.txt")

st.divider()

st.subheader("🖼️ Image Generation")
user_image_prompt = st.text_area(
    "Enter a prompt for image generation:",
    value="A warm, modern smart lamp named 'Lumi' on a wooden bedside table, soft golden light, minimalist interior."
)
if st.button("Generate Image(s)"):
    with st.spinner("Generating image(s)..."):
        images = generate_images(user_image_prompt, model=image_model, size=image_size, n=n_images)
    if images:
        st.success(f"Generated {len(images)} image(s)")
        cols = st.columns(len(images))
        for i, img_bytes in enumerate(images):
            img = Image.open(io.BytesIO(img_bytes))
            with cols[i]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
                st.download_button(
                    label=f"Download Image {i+1}",
                    data=img_bytes,
                    file_name=f"generated_{i+1}.png",
                    mime="image/png"
                )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and OpenAI API")
