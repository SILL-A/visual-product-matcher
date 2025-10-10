import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# ---- Load Data ----
@st.cache_resource
def load_data():
    df = pd.read_csv("data/fashion_with_embeddings.csv")
    embs = np.load("data/image_embeddings.npy")
    return df, embs

df, embeddings = load_data()

# ---- Load CLIP ----
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_clip()

# ---- Helper Functions ----
def get_image(p):
    if str(p).startswith("http"):
        return Image.open(BytesIO(requests.get(p, timeout=10).content)).convert("RGB")
    return Image.open(p).convert("RGB")

def get_emb(pil):
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        return model.get_image_features(**inputs).cpu().numpy().flatten()

def find_similar(q_emb, topk=5):
    sims = cosine_similarity([q_emb], embeddings)[0]
    idx = sims.argsort()[-topk:][::-1]
    return idx, sims[idx]

# ---- Streamlit UI ----
st.title("👗 Visual Product Matcher")
st.write("Upload an image or paste an image URL to find visually similar products.")

file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
url  = st.text_input("Or paste image URL (http...)")
topk = st.slider("Number of similar items", 3, 10, 5)
search = st.button("Find Similar Products")

if search:
    if file:
        image = Image.open(file).convert("RGB")
    elif url:
        image = get_image(url)
    else:
        st.warning("Please upload or paste an image.")
        st.stop()

    st.image(image, caption="Query Image", use_column_width=True)

    with st.spinner("Finding similar items..."):
        q_emb = get_emb(image)
        idx, scores = find_similar(q_emb, topk)

    st.subheader("Results")
    cols = st.columns(5)
    for i, (idn, sc) in enumerate(zip(idx, scores)):
        with cols[i % 5]:
            path = df.iloc[idn].get("ImageURL", df.iloc[idn].get("image_path",""))
            img = get_image(path)
            st.image(img, use_column_width=True)
            st.caption(f"{df.iloc[idn]['ProductTitle']}  \nScore: {sc:.2f}")
