import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ----------------------------------------------------
# PAGE CONFIG & THEME
# ----------------------------------------------------
st.set_page_config(page_title="AI Fashion Finder 👗", page_icon="🪞", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
/* global page */
.main {
    background: radial-gradient(circle at top left, #0d1117 0%, #161b22 100%);
    color: #f8f9fa;
    font-family: 'Poppins', sans-serif;
}
/* title gradient */
.title-text {
    background: -webkit-linear-gradient(90deg,#06beb6,#48b1bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: -10px;
}
/* subtitle */
.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 1.1rem;
}
/* buttons */
.stButton>button {
    background: linear-gradient(90deg,#06beb6,#48b1bf);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    padding: 0.75em 2em;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#48b1bf,#06beb6);
    transform: scale(1.02);
}
/* image cards */
.image-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
    margin: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    transition: transform 0.2s ease-in-out;
}
.image-card:hover {
    transform: scale(1.03);
}
/* captions */
.caption {
    font-size: 0.9rem;
    color: #cbd5e1;
    text-align: center;
}
/* reset container */
.reset-box {
    text-align:center;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE AREA
# ----------------------------------------------------
st.markdown("<h1 class='title-text'>AI Fashion Finder</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find visually similar fashion items powered by CLIP & deep learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------
# LOAD DATA & MODEL
# ----------------------------------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("data/fashion_with_embeddings.csv")
    embs = np.load("data/image_embeddings.npy")
    return df, embs

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

df, embeddings = load_data()
model, processor, device = load_clip()

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------
def get_image(source):
    try:
        if str(source).startswith("http"):
            return Image.open(BytesIO(requests.get(source, timeout=10).content)).convert("RGB")
        return Image.open(source).convert("RGB")
    except Exception:
        return None

def get_emb(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return model.get_image_features(**inputs).cpu().numpy().flatten()

def find_similar(q_emb, topk=5):
    sims = cosine_similarity([q_emb], embeddings)[0]
    idx = sims.argsort()[-topk:][::-1]
    return idx, sims[idx]

# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.header("Search Controls")
uploaded = st.sidebar.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])
url = st.sidebar.text_input("🔗 Or Paste Image URL")
topk = st.sidebar.slider("Number of Similar Results", 3, 12, 6)
category_filter = st.sidebar.selectbox("Filter by Category (optional)", ["All"] + sorted(df['Category'].unique()))
search = st.sidebar.button("🔍 Search")
reset = st.sidebar.button("🔁 Reset Search")

# ----------------------------------------------------
# RESET FUNCTION
# ----------------------------------------------------
if reset:
    st.experimental_rerun()

# ----------------------------------------------------
# MAIN SEARCH AREA
# ----------------------------------------------------
if search:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    elif url:
        img = get_image(url)
    else:
        st.warning("Please upload or paste an image.")
        st.stop()

    if img is None:
        st.error("Unable to load image. Please try another file or URL.")
        st.stop()

    # Display query image
    st.subheader("🎯 Query Image")
    cols = st.columns(3)
    with cols[1]:
        st.image(img, use_container_width=True, caption="Uploaded Item")

    # Compute similarity
    with st.spinner("✨ Analyzing your image..."):
        q_emb = get_emb(img)
        idx, scores = find_similar(q_emb, topk * 2)  # extra for filtering

    # Prepare results
    results = []
    for i, (idn, sc) in enumerate(zip(idx, scores)):
        product = df.iloc[idn]
        if category_filter != "All" and product.get("Category") != category_filter:
            continue
        results.append((product, sc))
        if len(results) >= topk:
            break

    # Display similar items
    st.markdown("---")
    st.subheader("💎 Similar Products")

    cols = st.columns(5)
    for i, (product, sc) in enumerate(results):
        title = product.get("ProductTitle", "Unknown Product")
        path = product.get("ImageURL", product.get("image_path", ""))
        prod_img = get_image(path)
        with cols[i % 5]:
            with st.container():
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                if prod_img:
                    st.image(prod_img, use_container_width=True)
                st.markdown(f"**{title[:40]}...**", unsafe_allow_html=True)
                st.markdown(f"<p class='caption'>Similarity: {sc:.2f}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='reset-box'><h3>👈 Start by uploading or pasting an image in the sidebar!</h3></div>", unsafe_allow_html=True)
