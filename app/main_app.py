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
        width: 60%;
        transition: 0.3s;
        margin: auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #019874;
        box-shadow: 0px 0px 10px #00b894;
    }
    .tip-box {
        background-color: rgba(30, 60, 90, 0.4);
        color: #a5d8ff;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 15px auto;
        text-align: center;
        width: 80%;
    }
    .caption {
        font-size: 14px;
        color: #d1d5db;
        margin-top: 5px;
        text-align: center;
    }
    img {
        border-radius: 10px !important;
        object-fit: contain !important;
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

# ------------------ CENTERED INPUT AREA ------------------
st.markdown("<div class='tip-box'>💡 Upload a clear picture of clothing, footwear, or accessories for best results.</div>", unsafe_allow_html=True)

# Center alignment for input area
center_col = st.columns([1, 2, 1])[1]
with center_col:
    uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    url = st.text_input("🔗 Or paste an image URL")
    topk = st.slider("Number of similar results", 3, 12, 6)
    
    c1, c2 = st.columns(2)
    with c1:
        search = st.button("🔍 Find Similar Products")
    with c2:
        reset = st.button("🔄 Reset")

# ------------------ RESET FUNCTION ------------------
if reset:
    st.experimental_rerun()

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

    # Reject non-fashion images (simple rule)
    possible_categories = ["apparel", "footwear", "clothing", "dress", "shirt", "pant", "top", "jean", "cap", "kurta", "t-shirt", "shoe", "sandal"]
    title_check = df["Category"].str.lower().unique().tolist()
    if not any(word in str(df["Category"].str.lower().tolist()) for word in possible_categories):
        st.warning("⚠️ Currently, you can only search for clothing or footwear items.")
        st.stop()

    with st.spinner("Analyzing image and finding similar products..."):
        q_emb = get_emb(img)
        idx, scores = find_similar(q_emb, topk)

    # Display results side by side
    st.markdown("---")
    st.subheader("👗 Query & Similar Products")

    left_col, right_col = st.columns([1, 3])

    # Query image
    with left_col:
        st.image(img, width=220, caption="Query Image")

    # Similar images
    with right_col:
        subcols = right_col.columns(min(5, len(idx)))
        for i, (idn, sc) in enumerate(zip(idx, scores)):
            product = df.iloc[idn]
            title = product.get("ProductTitle", "Unknown Product")
            path = product.get("ImageURL", product.get("image_path", ""))
            prod_img = get_image(path)

            with subcols[i % len(subcols)]:
                if prod_img:
                    st.image(prod_img, width=220)
                else:
                    st.image("https://via.placeholder.com/220?text=No+Image", width=220)
                st.markdown(f"**{title[:40]}...**")
                st.markdown(f"<p class='caption'>Similarity: {sc:.2f}</p>", unsafe_allow_html=True)
