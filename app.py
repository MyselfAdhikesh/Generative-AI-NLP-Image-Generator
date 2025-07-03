# Disable TensorFlow and Keras for compatibility
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load models only once
@st.cache_resource
def load_pipelines():
    # Load sentiment analysis
    sentiment = pipeline("sentiment-analysis")

    # Load English to French translation
    translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ml")


    # Load image generation (Stable Diffusion)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)

    return sentiment, translation, image_pipe

# Load models
sentiment_pipe, translation_pipe, image_pipe = load_pipelines()

# --- Streamlit UI ---
st.set_page_config(page_title="Generative AI App", layout="centered")
st.title("🧠 Generative AI: NLP + Image Generator")

# Sidebar for task selection
task = st.sidebar.radio("Select a Task", [
    "Sentiment Analysis",
    "English to French Translation",
    "Text-to-Image Generation"
])

# --- Sentiment Analysis ---
if task == "Sentiment Analysis":
    st.header("🧠 Sentiment Analysis")
    text_input = st.text_area("Enter a sentence:")
    if st.button("Analyze Sentiment") and text_input:
        result = sentiment_pipe(text_input)[0]
        st.success(f"Sentiment: **{result['label']}** (Confidence: {round(result['score'] * 100, 2)}%)")

# --- English to French Translation ---
elif task == "English to French Translation":
    st.header("🌍 English to French Translator")
    text_input = st.text_area("Enter English text:")
    if st.button("Translate") and text_input:
        result = translation_pipe(text_input)[0]
        st.success(f"French Translation: **{result['translation_text']}**")

# --- Image Generation ---
elif task == "Text-to-Image Generation":
    st.header("🎨 Text-to-Image Generator")
    prompt = st.text_input("Enter a prompt to generate an image:")
    if st.button("Generate Image") and prompt:
        with st.spinner("Generating image..."):
            image = image_pipe(prompt).images[0]
            image.save("generated_image.png")
            st.image(image, caption="Generated Image", use_column_width=True)
            with open("generated_image.png", "rb") as img_file:
                st.download_button("📥 Download Image", img_file, "generated_image.png", mime="image/png")
