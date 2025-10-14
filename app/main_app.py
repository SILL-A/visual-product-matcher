import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Visual Product Matcher", page_icon="🛍️", layout="wide")

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #f0f0f0;
        text-align: center;
    }
    .stButton>button {
        background-color: #00b894;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #019874;
    }
    .image-container {
        background-color: #161b22;
        border-radius: 15px;
        padding: 10px;
        margin: 10px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(255, 255, 255, 0.05);
    }
    .caption {
        font-size: 14px;
        color: #d1d5db;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🛍️ Visual Product Matcher")
st.markdown("<p style='text-align:center;'>Find visually similar fashion products using AI-powered image matching.</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ LOAD DATA ------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("data/fashion_with_embeddings.csv")
    embs = np.load("data/image_embeddings.npy")
    return df, embs

df, embeddings = load_data()

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_clip()

# ------------------ FUNCTIONS ------------------
def get_image(source):
    try:
        if str(source).startswith("http"):
            img = Image.open(BytesIO(requests.get(source, timeout=10).content)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        return img
    except Exception:
        return None

def get_emb(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy().flatten()
    return emb

def find_similar(q_emb, topk=5):
    sims = cosine_similarity([q_emb], embeddings)[0]
    idx = sims.argsort()[-topk:][::-1]
    return idx, sims[idx]

# ------------------ INPUT AREA ------------------
col1, col2 = st.columns([1, 3])
with col1:
    uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    url = st.text_input("🔗 Or paste an image URL")
    topk = st.slider("Number of similar results", 3, 12, 6)
    search = st.button("🔍 Find Similar Products")

with col2:
    st.info("💡 Tip: Upload a clear picture of clothing, footwear, or accessories for best results.")

# ------------------ SEARCH FUNCTIONALITY ------------------
if search:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    elif url:
        img = get_image(url)
    else:
        st.warning("Please upload or paste an image.")
        st.stop()

    if img is None:
        st.error("Could not load image. Try another file or URL.")
        st.stop()

    # Query image
    st.subheader("🖼️ Query Image")
    cols = st.columns(3)
    with cols[1]:  # center the image
        st.image(img, use_container_width=True, caption="Uploaded Image")

    # Compute similarity
    with st.spinner("Analyzing image and finding similar products..."):
        q_emb = get_emb(img)
        idx, scores = find_similar(q_emb, topk)

    st.markdown("---")
    st.subheader("✨ Similar Products")

    # Display results neatly
    cols = st.columns(5)
    for i, (idn, sc) in enumerate(zip(idx, scores)):
        product = df.iloc[idn]
        title = product.get("ProductTitle", "Unknown Product")
        path = product.get("ImageURL", product.get("image_path", ""))
        prod_img = get_image(path)

        with cols[i % 5]:
            with st.container():
                if prod_img:
                    st.image(prod_img, use_container_width=True)
                st.markdown(f"**{title[:40]}...**")
                st.markdown(f"<p class='caption'>Similarity: {sc:.2f}</p>", unsafe_allow_html=True)
