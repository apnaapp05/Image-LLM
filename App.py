# app.py
"""
Streamlit app that:
 - sends a user prompt to an LLM to generate text
 - asks an image-generation endpoint to create 1..N images from a prompt
 - displays results (text + generated image(s))
Requirements: see requirements.txt
Environment:
  export OPENAI_API_KEY="sk-..."
Or set in Streamlit secrets: {"OPENAI_API_KEY": "sk-..."}
"""

import os
import io
import base64
from typing import Optional, List

import streamlit as st
from PIL import Image
import requests
import openai

# ---------- Configuration & helpers ----------
def get_api_key() -> Optional[str]:
    # priority: streamlit secrets -> env var
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    return api_key

OPENAI_API_KEY = get_api_key()
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_IMAGE_SIZE = "1024x1024"

def call_llm_system(prompt: str, system_prompt: Optional[str], model: str) -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set environment variable or Streamlit secrets.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.8,
        n=1,
    )

    # Defensive extraction
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        # fallback to string representation
        return str(resp)

def generate_images(prompt: str, model: str, size: str, n: int) -> List[bytes]:
    """
    Returns list of image bytes (PNG/JPEG) generated from the prompt.
    """
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set environment variable or Streamlit secrets.")

    # Some providers support n>1 directly. We'll request n and ask for b64_json for reliability.
    try:
        resp = openai.Image.create(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            response_format="b64_json"
        )

        images_bytes = []
        for item in resp.get("data", []):
            b64 = item.get("b64_json")
            if b64:
                images_bytes.append(base64.b64decode(b64))
            else:
                # fallback if response has a url
                url = item.get("url")
                if url:
                    r = requests.get(url)
                    r.raise_for_status()
                    images_bytes.append(r.content)
        return images_bytes

    except Exception as e:
        # Try a fallback flow if the provider returned urls or a different shape
        try:
            resp = openai.Image.create(model=model, prompt=prompt, size=size, n=n)
            images_bytes = []
            for item in resp.get("data", []):
                url = item.get("url")
                if url:
                    r = requests.get(url)
                    r.raise_for_status()
                    images_bytes.append(r.content)
            if images_bytes:
                return images_bytes
        except Exception:
            pass
        raise RuntimeError(f"Image generation failed: {e}")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="LLM + Image Generator", layout="wide")

st.title("LLM + Image Generator (Streamlit)")
st.markdown("Generate text with an LLM and create images from a prompt. Configure models & API key in the sidebar.")

with st.sidebar:
    st.header("Configuration")
    llm_model = st.text_input("LLM model", value=DEFAULT_LLM_MODEL)
    image_model = st.text_input("Image model", value=DEFAULT_IMAGE_MODEL)
    image_size = st.selectbox("Image size", ["256x256", "512x512", "1024x1024"], index=2)
    max_images = st.slider("Number of image variations", min_value=1, max_value=6, value=2)
    use_system_prompt = st.checkbox("Use system prompt for LLM", value=True)
    if use_system_prompt:
        system_prompt = st.text_area("System prompt (instructions for LLM)",
                                     value="You are a helpful assistant that writes concise creative captions and descriptions.",
                                     height=120)
    else:
        system_prompt = None

    st.markdown("---")
    st.markdown("**API key**")
    st.markdown("Set `OPENAI_API_KEY` in environment or place it in Streamlit secrets under `OPENAI_API_KEY`.")
    if OPENAI_API_KEY:
        st.success("API key detected (from env or secrets).")
    else:
        st.warning("No API key found. Set in environment or Streamlit secrets before using.")

st.subheader("Prompts")
col_a, col_b = st.columns(2)
with col_a:
    user_prompt = st.text_area("LLM prompt (what you want the LLM to write)", height=180,
                               value="Write a short, vivid product description for a cozy smart lamp called 'Lumi' targeted at urban professionals.")
    generate_text_btn = st.button("Generate Text (LLM)")
with col_b:
    image_prompt = st.text_area("Image prompt (describe the image you want)", height=180,
                                value="A warm, modern smart lamp named 'Lumi' on a wooden bedside table, soft golden light, minimalist apartment interior, photographic style.")
    n_images = st.number_input("Number of images to generate", min_value=1, max_value=6, value=max_images)
    generate_image_btn = st.button("Generate Image(s)")

st.divider()

# LLM generation
if generate_text_btn:
    try:
        with st.spinner("Calling LLM..."):
            llm_output = call_llm_system(user_prompt, system_prompt, llm_model)
        st.markdown("### LLM output")
        st.write(llm_output)
        # Offer download
        st.download_button("Download text as .txt", data=llm_output, file_name="llm_output.txt", mime="text/plain")
    except Exception as exc:
        st.error(f"LLM call failed: {exc}")

# Image generation
if generate_image_btn:
    try:
        with st.spinner("Generating image(s)..."):
            images = generate_images(image_prompt, image_model, image_size, int(n_images))
        if not images:
            st.error("No images were returned by the image API.")
        else:
            st.markdown(f"### Generated {len(images)} image(s)")
            # Display in columns
            cols = st.columns(len(images))
            for i, img_bytes in enumerate(images):
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                except Exception:
                    # try to force into PIL via requests fallback
                    img = None
                with cols[i]:
                    if img:
                        st.image(img, caption=f"Image #{i+1}", use_column_width=True)
                        # prepare download bytes as PNG
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(f"Download Image #{i+1} (PNG)", data=buf.getvalue(),
                                           file_name=f"generated_{i+1}.png", mime="image/png")
                    else:
                        # If PIL couldn't open, offer raw bytes download
                        st.write("Couldn't render image preview, but you can download raw bytes.")
                        st.download_button(f"Download raw Image #{i+1}", data=img_bytes,
                                           file_name=f"generated_{i+1}.bin", mime="application/octet-stream")

    except Exception as exc:
        st.error(f"Image generation failed: {exc}")

st.markdown("---")
st.markdown("**Tips**")
st.write(
    "- For photorealistic images mention camera style / lens and lighting. For illustrations mention the art style.\n"
    "- If your provider limits the number of images per request, reduce the 'Number of images' value.\n"
    "- Keep your system prompt short and focused for best LLM results.\n"
)

