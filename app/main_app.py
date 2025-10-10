# main_app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests, base64, os, urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Visual Product Matcher", page_icon="🛍️", layout="wide")

# ---------- STYLES ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
  color: #2e2e2e;
}

/* Animated title */
@keyframes typing {
  from { width: 0; }
  to { width: 100%; }
}
@keyframes blink {
  50% { border-color: transparent; }
}

.title {
  font-weight: 800;
  font-size: 44px;
  text-align: center;
  background: linear-gradient(90deg, #a1c4fd, #c2e9fb, #d4fc79, #96e6a1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  overflow: hidden;
  white-space: nowrap;
  border-right: .15em solid #96e6a1;
  animation: typing 3.2s steps(40, end), blink .75s step-end infinite;
}

.subtitle {
  text-align:center;
  color:#444;
  font-weight:400;
  margin-bottom:20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.65);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(0,0,0,0.05);
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, #a1c4fd, #c2e9fb);
  color: #222;
  border-radius: 10px;
  font-weight: 600;
  border: none;
  padding: 0.6em 1.2em;
  transition: all .2s ease-in-out;
}
.stButton>button:hover {
  transform: translateY(-3px);
  background: linear-gradient(90deg, #d4fc79, #96e6a1);
}

/* Cards */
.card {
  background: rgba(255,255,255,0.7);
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  padding: 10px;
  transition: all 0.2s ease-in-out;
}
.card:hover { transform: scale(1.03); }

.card-img {
  width: 100%;
  height: 220px;
  object-fit: contain;
  border-radius: 10px;
}

.query {
  display:flex; justify-content:center;
  margin-top:20px;
}

.meta {
  color:#555;
  font-size:14px;
  text-align:center;
  margin-top:6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='title'>Visual Product Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Discover visually similar products instantly — powered by AI</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- CACHES ----------
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

# ---------- UTILITIES ----------
def fetch_image(source):
    """Return PIL image and bytes from file or URL"""
    try:
        if isinstance(source, bytes):
            return Image.open(BytesIO(source)).convert("RGB"), source
        s = str(source).strip()
        if os.path.exists(s):
            with open(s, "rb") as f: b = f.read()
            return Image.open(BytesIO(b)).convert("RGB"), b
        if s.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(s, headers=headers, timeout=10)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img, r.content
    except Exception:
        return None, None
    return None, None

def img_to_emb(img, model, processor, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

def sim_search(q_emb, embs, k=6):
    sims = cosine_similarity([q_emb], embs)[0]
    idx = sims.argsort()[-k:][::-1]
    return idx, sims[idx]

def img_html(bts):
    return f"data:image/jpeg;base64,{base64.b64encode(bts).decode('utf-8')}"

# ---------- LOAD MODEL/DATA ----------
df, embeddings = load_data()
model, processor, device = load_clip()

# ---------- SIDEBAR ----------
st.sidebar.header("Search / Filters")
uploaded = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
url_input = st.sidebar.text_input("Or paste image URL (direct link)", placeholder="https://example.com/image.jpg", label_visibility="visible")
top_k = st.sidebar.slider("Results to show", 3, 12, 6)
category_options = ["All"] + sorted(df['Category'].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Filter category", category_options)
colA, colB = st.sidebar.columns(2)
btn_search = colA.button("Search")

# ---------- SEARCH ----------
if btn_search:
    src = None
    if uploaded:
        src = uploaded.read()
    elif url_input:
        src = url_input.strip()
    if not src:
        st.warning("Please upload or paste an image first.")
        st.stop()

    img, img_b = fetch_image(src)
    if img is None:
        st.error("Could not fetch image. Try another URL.")
        st.stop()

    st.markdown("<h4 style='text-align:center;'>Query Image</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='query'><div class='card'><img class='card-img' src='{img_html(img_b)}'></div></div>", unsafe_allow_html=True)

    with st.spinner("Finding similar items..."):
        q_emb = img_to_emb(img, model, processor, device)
        idxs, scores = sim_search(q_emb, embeddings, top_k*2)

    results = []
    for i, s in zip(idxs, scores):
        row = df.iloc[i]
        if category_filter != "All" and row["Category"] != category_filter:
            continue
        results.append((row, s))
        if len(results) >= top_k:
            break

    if not results:
        st.info("No similar items found for this image.")
    else:
        st.markdown("<h4 style='text-align:center;'>Similar Products</h4>", unsafe_allow_html=True)
        cols = st.columns(min(5, len(results)))
        for i, (r, sc) in enumerate(results):
            c = cols[i % 5]
            with c:
                pic, pbytes = fetch_image(r["ImageURL"])
                if pic is not None:
                    c.markdown(f"<div class='card'><img class='card-img' src='{img_html(pbytes)}'><div class='meta'><b>{r['ProductTitle'][:40]}</b><br>Score: {sc:.3f}</div></div>", unsafe_allow_html=True)
                else:
                    c.markdown(f"<div class='card'><div class='meta'>Image unavailable<br>Score: {sc:.3f}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:#666; margin-top:1em;'>Search complete — use Reset to start over.</div>", unsafe_allow_html=True)

else:
    st.markdown("<div style='text-align:center; color:#777;'>Upload an image or paste a URL in the sidebar to start.</div>", unsafe_allow_html=True)
