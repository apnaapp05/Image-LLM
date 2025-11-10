# app.py
"""
Streamlit app that:
 - sends a user prompt to an LLM to generate text
 - asks an image-generation endpoint to create an image from a prompt
 - displays results (text + generated image)
Requirements:
  pip install streamlit openai pillow requests
Environment:
  export OPENAI_API_KEY="sk-..."
Notes:
 - This example uses the openai package (classic pattern). Your provider may have a different API.
 - Do NOT hardcode your API key.
"""

import os
import io
import base64
from typing import Optional

import streamlit as st
from PIL import Image
import requests
import openai

# ---------- Configuration ----------
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Optional: choose model names you prefer
LLM_MODEL = "gpt-4o"            # replace with your preferred chat model
IMAGE_SIZE = "1024x1024"       # "256x256", "512x512", "1024x1024"
IMAGE_MODEL = "gpt-image-1"    # replace with your provider's image model name if needed

# ---------- Helper functions ----------
def call_llm_system(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call a chat/completion LLM and return the assistant reply as text.
    This uses the OpenAI ChatCompletion API shape; adapt if your SDK differs.
    """
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Synchronous call; change to async if you prefer
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.8,
        n=1,
    )
    # defensive: extract text carefully
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return str(resp)

def generate_image_from_prompt(prompt: str, size: str = IMAGE_SIZE) -> bytes:
    """
    Generate an image from a prompt. Returns raw bytes of the image (PNG/JPEG).
    This assumes OpenAI-style images API which may return a base64 string or a url.
    Adapt if your provider returns different payloads.
    """
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Try the 'images' endpoint: some SDKs use openai.Image.create
    try:
        resp = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size,
            n=1,
            response_format="b64_json"  # ask for base64 so we get bytes reliably
        )
        b64 = resp["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64)
        return image_bytes
    except Exception as e:
        # Fallback: if API returns a URL, fetch it
        try:
            resp = openai.Image.create(model=IMAGE_MODEL, prompt=prompt, size=size, n=1)
            url = resp["data"][0]["url"]
            r = requests.get(url)
            r.raise_for_status()
            return r.content
        except Exception as e2:
            raise RuntimeError(f"Image generation failed: {e} / {e2}")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="LLM + Image Generator", layout="centered")

st.title("LLM + Image Generator (Streamlit)")

with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("LLM model", value=LLM_MODEL)
    image_model = st.text_input("Image model", value=IMAGE_MODEL)
    image_size = st.selectbox("Image size", ["256x256", "512x512", "1024x1024"], index=2)
    enable_system_prompt = st.checkbox("Use system prompt (for LLM)", value=True)
    if enable_system_prompt:
        system_prompt = st.text_area("System prompt (instructions for the LLM)",
                                     value="You are a helpful assistant that writes concise creative captions and descriptions.")
    else:
        system_prompt = None

st.subheader("User prompts")
user_prompt = st.text_area("Describe what you want the LLM to produce (text).", height=120,
                           value="Write a short, vivid product description for a cozy smart lamp called 'Lumi' targeted at urban professionals.")
image_prompt = st.text_area("Describe the image you want generated.", height=140,
                            value="A warm, modern smart lamp named 'Lumi' on a wooden bedside table, soft golden light, minimalist apartment interior, photographic style.")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate Text (LLM)"):
        try:
            with st.spinner("Calling LLM..."):
                # Update globals if user changed in sidebar
                resp_text = call_llm_system(user_prompt, system_prompt)
            st.success("Text generated")
            st.markdown("**LLM output:**")
            st.write(resp_text)
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")

with col2:
    if st.button("Generate Image"):
        try:
            with st.spinner("Generating image..."):
                image_bytes = generate_image_from_prompt(image_prompt, size=image_size)
            st.success("Image generated")
            # Display bytes as image
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Generated image", use_column_width=True)

            # Offer download button
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("Download PNG", data=byte_im, file_name="generated.png", mime="image/png")
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")

st.divider()
st.markdown("### Tips")
st.write(
    "- Use a clear, descriptive prompt for images (subject, environment, style, mood, camera settings if you want photorealism).\n"
    "- Keep your system prompt focused and short for best LLM results.\n"
    "- If your provider returns URLs rather than base64 image data, the fallback fetch will attempt to download it."
)
