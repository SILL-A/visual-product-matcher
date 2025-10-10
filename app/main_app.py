# main_app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests, base64, os, urllib.parse, traceback
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Visual Product Matcher ✨", page_icon="🛍️", layout="wide")

# ---------- STYLES (premium UI) ----------
st.markdown("""
<style>
:root {
  --accent1: #06beb6;
  --accent2: #48b1bf;
  --glass: rgba(255,255,255,0.04);
  --muted: #b9c2c9;
}
body { background: radial-gradient(circle at 10% 10%, #0d1117 0%, #15171b 60%); }

/* Title */
.title {
  font-family: 'Poppins', sans-serif;
  font-weight: 800;
  font-size: 42px;
  text-align: center;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}
.subtitle { text-align:center; color:var(--muted); margin-bottom:18px; }

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,var(--accent1),var(--accent2));
  color: white;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.55rem 1rem;
  transition: transform .12s ease-in-out, box-shadow .12s;
  box-shadow: 0 6px 18px rgba(8,8,8,0.3);
}
.stButton>button:hover { transform: translateY(-3px); }

/* Cards / images */
.card {
  background: rgba(255,255,255,0.03);
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 8px 20px rgba(2,6,23,0.45);
  transition: transform 0.16s ease, box-shadow 0.16s;
}
.card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 18px 40px rgba(2,6,23,0.55); }

.card-img {
  width: 100%;
  height: 220px;
  object-fit: contain;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(0,0,0,0.02));
}

/* small text */
.meta { color: var(--muted); font-size: 13px; margin-top:6px; }

/* center query image container */
.query-wrap { display:flex; justify-content:center; align-items:center; padding:8px; }

/* fallback placeholder */
.placeholder {
  width:100%;
  height:220px;
  border-radius:10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
  display:flex;
  align-items:center;
  justify-content:center;
  color: #7f8a94;
  font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='title'>AI Fashion Finder</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Premium visual search — upload an image or paste an image URL to find similar products.</div>", unsafe_allow_html=True)
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

# ---------- IMAGE FETCH UTILITIES ----------
def try_fix_url(u):
    # common quick fixes: add https, remove query params
    parsed = urllib.parse.urlparse(u)
    if not parsed.scheme:
        u2 = "https://" + u.lstrip("/")
        return u2
    if parsed.scheme == "http":
        u2 = u.replace("http://", "https://", 1)
        if u2 != u:
            return u2
    # strip query
    if parsed.query:
        u2 = urllib.parse.urlunparse(parsed._replace(query=""))
        return u2
    return u

def fetch_image_bytes(source, timeout=12):
    """
    source: can be:
      - bytes (uploaded file.read())
      - local file path string
      - url string
    returns: (bytes, content_type) or (None, None) if failed
    """
    # uploaded bytes
    if isinstance(source, (bytes, bytearray)):
        # attempt to infer type
        return bytes(source), "image/jpeg"

    s = str(source).strip()
    # local file
    if os.path.exists(s):
        try:
            with open(s, "rb") as f:
                data = f.read()
            # try to detect format via PIL
            try:
                img = Image.open(BytesIO(data))
                ct = f"image/{img.format.lower()}"
            except Exception:
                ct = "image/jpeg"
            return data, ct
        except Exception:
            return None, None

    # if it looks like a URL
    if s.startswith("http") or s.startswith("www.") or ("//" in s and "." in s):
        tried = set()
        candidates = [s]
        # add some fixes candidate
        try:
            candidates.append(try_fix_url(s))
        except Exception:
            pass

        for url in candidates:
            if not url or url in tried:
                continue
            tried.add(url)
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
                if resp.status_code != 200:
                    continue
                content_type = resp.headers.get("content-type", "")
                if "image" not in content_type:
                    # sometimes servers omit content-type; still try
                    pass
                data = resp.content
                # test with PIL
                try:
                    Image.open(BytesIO(data)).verify()
                except Exception:
                    # invalid image bytes
                    continue
                ct = content_type.split(";")[0] if content_type else "image/jpeg"
                return data, ct
            except Exception:
                continue
    return None, None

def bytes_to_data_uri(bts, content_type="image/jpeg"):
    b64 = base64.b64encode(bts).decode("utf-8")
    return f"data:{content_type};base64,{b64}"

# ---------- EMBEDDING / SIMILARITY ----------
def image_to_embedding_from_bytes(bts, model, processor, device):
    try:
        img = Image.open(BytesIO(bts)).convert("RGB")
    except Exception as e:
        raise RuntimeError("Invalid image bytes") from e
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

def find_similar_indices(q_emb, all_embs, top_k=6):
    sims = cosine_similarity([q_emb], all_embs)[0]
    idx = sims.argsort()[-top_k:][::-1]
    return idx, sims[idx]

# ---------- LOAD DATA & MODEL ----------
with st.spinner("Loading dataset and model (cached after first load)..."):
    df, embeddings = load_data()
    model, processor, device = load_clip()

# ---------- SIDEBAR ----------
st.sidebar.header("Search / Filters")
uploaded = st.sidebar.file_uploader("Upload image", type=["jpg","jpeg","png"])
url_input = st.sidebar.text_input("Or paste image URL (direct link)")
top_k = st.sidebar.slider("Results to show", 3, 12, 6)
category_options = ["All"] + sorted(df['Category'].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Filter category", category_options)
btn_search = st.sidebar.button("Search")
btn_reset = st.sidebar.button("Reset")

if btn_reset:
    st.experimental_rerun()

# ---------- MAIN ACTION ----------
if btn_search:
    # get bytes robustly
    source = None
    if uploaded:
        source = uploaded.read()
    elif url_input:
        source = url_input.strip()

    if not source:
        st.warning("Upload an image file or paste an image URL in the sidebar.")
        st.stop()

    # fetch bytes
    bts, ctype = fetch_image_bytes(source)
    if bts is None:
        st.error("Could not fetch the image. If you pasted a URL, try opening the image in a browser and copy the direct image link (ends with .jpg/.png).")
        st.stop()

    # show query image centered and same size as results
    st.subheader("Query Image")
    b64 = bytes_to_data_uri(bts, ctype)
    query_html = f"""<div class="query-wrap card"><img src="{b64}" class="card-img" /></div>"""
    st.markdown(query_html, unsafe_allow_html=True)

    # compute embedding
    with st.spinner("Computing embedding & searching..."):
        try:
            q_emb = image_to_embedding_from_bytes(bts, model, processor, device)
        except Exception as e:
            st.error("Failed to compute embedding for the given image. Try another image.")
            st.stop()

        # search a bit more than top_k to apply category filter
        idxs, scores = find_similar_indices(q_emb, embeddings, top_k=top_k*3)

    # collect final results with optional filter
    results = []
    for idx, sc in zip(idxs, scores):
        row = df.iloc[idx]
        if category_filter != "All" and row.get("Category") != category_filter:
            continue
        results.append((row, float(sc)))
        if len(results) >= top_k:
            break

    if not results:
        st.info("No similar products found (try a different image or lower filters).")
    else:
        st.markdown("---")
        st.subheader("Similar Products")
        # create columns based on number of results (max 5)
        ncols = min(5, len(results))
        cols = st.columns(ncols)
        for i, (row, score) in enumerate(results):
            with cols[i % ncols]:
                # load product image bytes
                candidate_src = row.get("ImageURL") or row.get("image_path") or row.get("Image")
                bts_c, ctype_c = fetch_image_bytes(candidate_src)
                if bts_c is None:
                    card_html = f"""<div class="card"><div class="placeholder">Image not available</div><div class="meta"> {row.get('ProductTitle','')}<br/><small>Score: {score:.3f}</small></div></div>"""
                    st.markdown(card_html, unsafe_allow_html=True)
                else:
                    b64c = bytes_to_data_uri(bts_c, ctype_c)
                    card_html = f"""
                    <div class="card">
                      <img src="{b64c}" class="card-img"/>
                      <div class="meta"><strong>{row.get('ProductTitle','')[:60]}</strong><br/>Score: {score:.3f}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='text-align:center; color:#a6b0b6;'>Tip: Click Reset in the sidebar to start a new search.</div>", unsafe_allow_html=True)

else:
    # initial empty state
    st.markdown("<div style='text-align:center; margin-top:40px; color:#a6b0b6;'>Upload an image or paste an image URL in the sidebar to begin. Try a clear product photo (white or neutral background works best).</div>", unsafe_allow_html=True)
