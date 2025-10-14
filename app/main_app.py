# main_app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests, os, urllib.parse, base64
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ------------------ PAGE ------------------
st.set_page_config(page_title="Visual Product Matcher", page_icon="🛍️", layout="wide")

# ------------------ STYLES ------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #e8eef5;
        background: radial-gradient(circle at top left, #1f1f1f, #0d1117);
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        background: linear-gradient(90deg, #a1c4fd, #c2e9fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #b8c7d9;
        margin-bottom: 30px;
    }

    .center-box {
        width: 780px;
        margin: 0 auto;
        padding: 28px 30px 34px 30px;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.05);
    }

    .tip {
        text-align: center;
        color: #b8e6ff;
        background: rgba(34,193,195,0.1);
        border-left: 4px solid #22c1c3;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 14px;
    }

    .stButton>button {
        display: block;
        margin: 10px auto 0 auto;
        width: 320px;
        height: 46px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        color: #0d1117;
        background: linear-gradient(90deg, #22c1c3, #fdbb2d);
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(253,187,45,0.25);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 35px rgba(253,187,45,0.35);
    }

    .uniform-img {
        width: 220px;
        height: 220px;
        object-fit: contain;
        border-radius: 14px;
        background: rgba(255,255,255,0.04);
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.3s ease;
    }
    .uniform-img:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(255,255,255,0.1);
    }

    .meta {
        text-align: center;
        color: #b9c9d4;
        margin-top: 8px;
        font-size: 13px;
    }

    .footer-tip {
        text-align:center;
        font-size:13px;
        color:#8b9cad;
        margin-top:25px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<div class='title'>Visual Product Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find visually similar clothing and footwear items instantly 👗👟</div>", unsafe_allow_html=True)

# ------------------ LOAD DATA & MODEL ------------------
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

# ------------------ HELPERS ------------------
def try_fix_url(u):
    parsed = urllib.parse.urlparse(u)
    if not parsed.scheme:
        return "https://" + u.lstrip("/")
    if parsed.scheme == "http":
        return u.replace("http://", "https://", 1)
    if parsed.query:
        return urllib.parse.urlunparse(parsed._replace(query=""))
    return u

def fetch_image_bytes(source, timeout=10):
    if not source:
        return None, None
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "image/jpeg"
    s = str(source).strip()
    if os.path.exists(s):
        with open(s, "rb") as f:
            return f.read(), "image/jpeg"
    for url in [s, try_fix_url(s)]:
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
            if r.status_code == 200:
                return r.content, r.headers.get("content-type","image/jpeg")
        except Exception:
            continue
    return None, None

def get_emb(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

def find_similar(q_emb, topk=5):
    sims = cosine_similarity([q_emb], embeddings)[0]
    idx = sims.argsort()[-topk:][::-1]
    return idx, sims[idx]

# ------------------ SMART INPUT HANDLER ------------------
st.markdown("<div class='center-box'>", unsafe_allow_html=True)
st.markdown("<div class='tip'>💡 Upload a clear picture of clothing, footwear, or accessories for best results.</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
    url_input = st.text_input("Or paste an image URL")

    # 🧠 Auto-clear logic
    if "last_input_type" not in st.session_state:
        st.session_state.last_input_type = None

    if uploaded is not None and st.session_state.last_input_type != "upload":
        url_input = ""
        st.session_state.last_input_type = "upload"

    elif url_input and st.session_state.last_input_type != "url":
        uploaded = None
        st.session_state.last_input_type = "url"
        st.info("🔄 Switched to URL input (previous upload cleared).")

    top_k = st.slider("Number of similar results", 3, 12, 6)
    search_clicked = st.button("🔍 Find Similar Products")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ SEARCH LOGIC ------------------
if search_clicked:
    source = uploaded.read() if uploaded else url_input
    if not source:
        st.warning("Please upload or paste an image to search.")
        st.stop()

    image_bytes, ctype = fetch_image_bytes(source)
    if image_bytes is None:
        st.error("Could not load image. Please try another file or URL.")
        st.stop()

    q_emb = get_emb(image_bytes)
    idx, sims = find_similar(q_emb, top_k)

    # Fashion check
    allowed = {"Apparel", "Footwear"}
    final = [(df.iloc[i], sims[i]) for i in idx if df.iloc[i]["Category"] in allowed]
    if not final:
        st.error("⚠️ We currently support clothing and footwear only.")
        st.stop()

    st.markdown("---")
    left, right = st.columns([1, 3])

    with left:
        st.markdown("<h5 style='text-align:center;'>Query Image</h5>", unsafe_allow_html=True)
        st.image(Image.open(BytesIO(image_bytes)), use_container_width=True)

    with right:
        st.markdown("<h5 style='text-align:center;'>Similar Products</h5>", unsafe_allow_html=True)
        cols = st.columns(min(5, len(final)))
        for i, (prod, sc) in enumerate(final):
            img_url = prod.get("ImageURL", "")
            bytes_res, ct_res = fetch_image_bytes(img_url)
            if bytes_res:
                img_data = base64.b64encode(bytes_res).decode()
                cols[i % 5].markdown(
                    f"<img src='data:{ct_res};base64,{img_data}' class='uniform-img'/>"
                    f"<div class='meta'><b>{prod['ProductTitle'][:40]}</b><br/>Score: {sc:.2f}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div class='footer-tip'>Tip: choose a neutral background image for best match quality.</div>", unsafe_allow_html=True)
