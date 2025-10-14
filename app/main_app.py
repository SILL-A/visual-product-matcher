import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Visual Product Matcher", page_icon="🛍️", layout="wide")

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ Visual Product Matcher")
st.markdown("Find **visually similar fashion products** by uploading or pasting an image URL below.")

# ---------------------- LOAD DATA ----------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("data/fashion_with_embeddings.csv")
    embs = np.load("data/image_embeddings.npy")
    return df, embs

df, embeddings = load_data()

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_clip()

# ---------------------- HELPER FUNCTIONS ----------------------
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

# ---------------------- UI INPUTS ----------------------
col1, col2 = st.columns([1, 3])
with col1:
    uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    url = st.text_input("🔗 Or paste an image URL")
    topk = st.slider("Results to show", 3, 12, 6)
    search = st.button("🔍 Find Similar Products")

with col2:
    st.info("💡 Tip: Upload a clear image of a fashion item (e.g., shirt, shoe, dress).")

# ---------------------- SEARCH ----------------------
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

    st.subheader("🖼️ Query Image")
    st.image(img, use_container_width=True)

    with st.spinner("Extracting features and finding similar items..."):
        q_emb = get_emb(img)
        idx, scores = find_similar(q_emb, topk)

    st.subheader("✨ Similar Products")
    cols = st.columns(5)

    for i, (idn, sc) in enumerate(zip(idx, scores)):
        product = df.iloc[idn]
        title = product.get("ProductTitle", "Unknown Product")
        path = product.get("ImageURL", product.get("image_path", ""))
        prod_img = get_image(path)

        with cols[i % 5]:
            if prod_img:
                st.image(prod_img, use_container_width=True)
            st.markdown(f"**{title}**")
            st.caption(f"Similarity: {sc:.2f}")
