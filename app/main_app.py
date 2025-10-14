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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    body { background: linear-gradient(180deg,#0b0f12 0%, #0f1417 100%); color: #e6eef3; }

    .title { font-size:34px; font-weight:700; text-align:center;
             background: linear-gradient(90deg,#a1c4fd,#c2e9fb);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:6px; }
    .subtitle{ text-align:center; color:#9fb3c8; margin-bottom:18px; }

    /* centered input area */
    .center-box { width: 760px; margin: 0 auto 18px auto; padding: 18px; border-radius: 12px;
                  background: rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }

    .tip { background: rgba(16,40,60,0.6); color:#bfe6ff; padding:10px 14px; border-radius:8px; text-align:center; margin-bottom:12px; }

    /* single centered button */
    .search-btn {
      display:block;
      margin: 16px auto 0 auto;
      width: 320px;
      height:44px;
      border-radius:10px;
      background: linear-gradient(90deg,#22c1c3,#fdbb2d);
      color: #072026;
      font-weight:700;
      font-size:16px;
      border:none;
      box-shadow: 0 10px 30px rgba(34,193,195,0.12);
      cursor:pointer;
    }
    .search-btn:hover { transform: translateY(-3px); box-shadow: 0 14px 42px rgba(34,193,195,0.18); }

    /* uniform images */
    .uniform-img { width:220px; height:220px; object-fit:contain; border-radius:10px; display:block; margin: 0 auto; }

    .meta { text-align:center; color:#b9c9d4; margin-top:8px; font-size:13px; }

    /* smaller labels */
    label { font-size:14px; color:#c8d7e0; }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<div class='title'>Visual Product Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find visually similar clothing and footwear items instantly.</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ DATA & MODEL ------------------
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

with st.spinner("Loading dataset and model (cached after first run)..."):
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
    if source is None or (isinstance(source, str) and source.strip() == ""):
        return None, None
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "image/jpeg"
    s = str(source).strip()
    if os.path.exists(s):
        try:
            with open(s, "rb") as f:
                return f.read(), "image/jpeg"
        except Exception:
            return None, None
    # try URL and small fixes
    candidates = [s]
    try:
        candidates.append(try_fix_url(s))
    except Exception:
        pass
    headers = {"User-Agent":"Mozilla/5.0"}
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if r.status_code != 200:
                continue
            data = r.content
            # quick validation
            try:
                Image.open(BytesIO(data)).verify()
            except Exception:
                continue
            ctype = r.headers.get("content-type","image/jpeg").split(";")[0]
            return data, ctype
        except Exception:
            continue
    return None, None

def img_bytes_to_datauri(bts, ctype="image/jpeg"):
    return f"data:{ctype};base64,{base64.b64encode(bts).decode()}"

def image_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def emb_from_bytes(bts):
    img = image_from_bytes(bts)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

def find_similar(q_emb, all_embs, topk=6):
    sims = cosine_similarity([q_emb], all_embs)[0]
    idx = sims.argsort()[-topk:][::-1]
    return idx, sims[idx]

# ------------------ CENTERED INPUT UI ------------------
st.markdown("<div class='center-box'>", unsafe_allow_html=True)
st.markdown("<div class='tip'>💡 Upload a clear picture of clothing, footwear, or accessories for best results.</div>", unsafe_allow_html=True)

# center content inside the box
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    url_input = st.text_input("Or paste an image URL (direct link)")
    top_k = st.slider("Number of similar results", min_value=3, max_value=12, value=6)
    # single centered button (uses custom CSS class)
    search_clicked = st.button("🔍  Find Similar Products", key="search")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ SEARCH LOGIC ------------------
if search_clicked:
    source = None
    if uploaded is not None:
        try:
            source = uploaded.read()
        except Exception:
            source = None
    elif url_input and url_input.strip():
        source = url_input.strip()

    if not source:
        st.warning("Please upload a file or paste a direct image URL, then click the button.")
        st.stop()

    bts, ctype = fetch_image_bytes(source)
    if bts is None:
        st.error("Could not fetch the image. If using a URL, make sure it's a direct image link (ends with .jpg/.png).")
        st.stop()

    # show query image left and results right
    st.markdown("---")
    left_col, right_col = st.columns([1, 3])

    # compute embedding
    with st.spinner("Computing embedding and searching..."):
        try:
            q_emb = emb_from_bytes(bts)
        except Exception:
            st.error("Failed to compute embedding for the image. Try a different image.")
            st.stop()
        idxs, sims = find_similar(q_emb, embeddings, topk=top_k*3)

    # quick fashion-check: if most results not apparel/footwear, warn
    allowed = {"Apparel", "Footwear"}
    final = []
    for idx, sc in zip(idxs, sims):
        row = df.iloc[idx]
        final.append((row, float(sc)))
        if len(final) >= top_k:
            break
    allowed_count = sum(1 for (r, s) in final if str(r.get("Category","")).strip() in allowed)
    avg_score = np.mean([s for (r,s) in final]) if final else 0.0
    if allowed_count < max(1, top_k//2) or avg_score < 0.22:
        st.error("We currently support search for clothing and footwear only. The uploaded image doesn't look like a clothing/footwear item. Try another image.")
        st.stop()

    # left: query image (uniform size)
    with left_col:
        data_uri = img_bytes_to_datauri(bts, ctype)
        st.markdown(f"<img src='{data_uri}' class='uniform-img'/>", unsafe_allow_html=True)
        st.markdown("<div class='meta'><b>Query Image</b></div>", unsafe_allow_html=True)

    # right: grid of uniform result images (220x220)
    with right_col:
        ncols = min(5, len(final))
        cols = st.columns(ncols)
        for i, (row, score) in enumerate(final):
            with cols[i % ncols]:
                candidate = row.get("ImageURL") or row.get("image_path") or row.get("Image")
                bts_c, ctype_c = fetch_image_bytes(candidate)
                if bts_c is None:
                    # fallback placeholder
                    st.markdown(f"<img src='https://via.placeholder.com/220?text=No+Image' class='uniform-img'/>", unsafe_allow_html=True)
                else:
                    data_uri_c = img_bytes_to_datauri(bts_c, ctype_c)
                    st.markdown(f"<img src='{data_uri_c}' class='uniform-img'/>", unsafe_allow_html=True)
                st.markdown(f"<div class='meta'><b>{row.get('ProductTitle','')[:54]}</b><br/>Score: {score:.3f}</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; color:#99a6b3; margin-top:12px;'>Tip: choose a clear product photo for best matches.</div>", unsafe_allow_html=True)
