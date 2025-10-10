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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Visual Product Matcher", page_icon="🛍️", layout="wide")

# ---------------- CSS (pastel / premium) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
body {
  background: linear-gradient(135deg, #f7f8fc 0%, #f0f7f9 30%, #fdf7fb 100%);
  color: #1f2937;
}

/* Title */
.title {
  font-weight: 800;
  font-size: 42px;
  text-align: center;
  background: linear-gradient(90deg,#cfd9ff,#e4fbff,#ffd6e0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 4px;
}
.subtitle { text-align:center; color:#475569; margin-bottom:18px; }

/* Sidebar look */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(6px);
  border-right: 1px solid rgba(0,0,0,0.03);
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#cfd9ff,#e4fbff);
  color: #073b4c;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.6em 1.2em;
  transition: transform .14s ease;
  box-shadow: 0 8px 20px rgba(14,30,37,0.06);
}
.stButton>button:hover { transform: translateY(-3px); }

/* Card */
.card {
  background: rgba(255,255,255,0.85);
  border-radius: 14px;
  box-shadow: 0 12px 28px rgba(14,30,37,0.08);
  padding: 10px;
  transition: transform .14s ease;
}
.card:hover { transform: translateY(-6px); }

/* Image style - consistent size */
.card-img {
  width:100%;
  height: 260px;
  object-fit: contain;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(0,0,0,0.02), rgba(255,255,255,0.02));
}

/* small text */
.meta { color:#475569; font-size:14px; text-align:center; margin-top:8px; }

/* Query area */
.query { display:flex; justify-content:center; margin-top:18px; }
.placeholder { width:100%; height:260px; border-radius:10px; display:flex;align-items:center;justify-content:center;color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>Visual Product Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find visually similar clothing and footwear items instantly.</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- CACHES (data & model) ----------------
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

# ---------------- UTILITIES ----------------
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
    # source: bytes or local path or url string
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "image/jpeg"
    s = str(source).strip()
    if not s:
        return None, None
    # local path
    if os.path.exists(s):
        try:
            with open(s, "rb") as f:
                data = f.read()
            return data, "image/jpeg"
        except Exception:
            return None, None
    # try URL candidates
    candidates = [s]
    try:
        candidates.append(try_fix_url(s))
    except Exception:
        pass
    for url in candidates:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if r.status_code != 200:
                continue
            data = r.content
            # validate with PIL
            try:
                Image.open(BytesIO(data)).verify()
            except Exception:
                continue
            ctype = r.headers.get("content-type", "image/jpeg").split(";")[0]
            return data, ctype
        except Exception:
            continue
    return None, None

def bytes_to_data_uri(bts, ctype="image/jpeg"):
    return f"data:{ctype};base64,{base64.b64encode(bts).decode()}"

def image_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def img_to_embedding_from_bytes(bts, model, processor, device):
    img = image_from_bytes(bts)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

def find_similar_indices(q_emb, all_embs, top_k=6):
    sims = cosine_similarity([q_emb], all_embs)[0]
    idx = sims.argsort()[-top_k:][::-1]
    return idx, sims[idx]

# ---------------- LOAD DATA & MODEL ----------------
with st.spinner("Loading dataset and model (cached after first run)..."):
    df, embeddings = load_data()
    model, processor, device = load_clip()

# ---------------- SIDEBAR (no filter) ----------------
st.sidebar.header("Search / Controls")
# file uploader key 'uploaded_file'
uploaded_file = st.sidebar.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"], key="uploaded_file")
# use text_area to avoid the 'press Enter to apply' hint
url_text = st.sidebar.text_area("Or paste image URL (direct link)", value=st.session_state.get("url_text",""), height=80, key="url_text")
top_k = st.sidebar.slider("Results to show", min_value=3, max_value=12, value=6)
search_btn = st.sidebar.button("Search")
reset_btn = st.sidebar.button("Reset")

# Reset behavior: clear uploader and url_text and rerun
if reset_btn:
    if "uploaded_file" in st.session_state:
        st.session_state["uploaded_file"] = None
    st.session_state["url_text"] = ""
    st.experimental_rerun()

# ---------------- SEARCH ACTION ----------------
if search_btn:
    # determine source (uploaded has priority)
    src = None
    if st.session_state.get("uploaded_file") is not None:
        # UploadedFile object -> read bytes
        up = st.session_state.get("uploaded_file")
        try:
            b = up.read()
            src = b
        except Exception:
            src = None
    elif st.session_state.get("url_text", "").strip():
        src = st.session_state.get("url_text").strip()

    if not src:
        st.warning("Please upload an image or paste a direct image URL in the sidebar, then click Search.")
        st.stop()

    # fetch bytes
    bts, ctype = fetch_image_bytes(src)
    if bts is None:
        st.error("Could not fetch the image. If you pasted a URL, make sure it's a direct image link (ends with .jpg/.png) and try again.")
        st.stop()

    # show query image centered (same size as results)
    st.subheader("Query Image")
    data_uri = bytes_to_data_uri(bts, ctype)
    st.markdown(f"<div class='query'><div class='card'><img src='{data_uri}' class='card-img'/></div></div>", unsafe_allow_html=True)

    # compute embedding and search
    with st.spinner("Computing embedding and searching..."):
        try:
            q_emb = img_to_embedding_from_bytes(bts, model, processor, device)
        except Exception:
            st.error("Failed to compute embedding for this image. Try a different image.")
            st.stop()

        idxs, scores = find_similar_indices(q_emb, embeddings, top_k=top_k*3)

    # quick check: do top-k results belong to Apparel/Footwear?
    allowed = {"Apparel", "Footwear"}
    final = []
    for idx, sc in zip(idxs, scores):
        row = df.iloc[idx]
        final.append((row, float(sc)))
        if len(final) >= top_k:
            break

    # validation: if less than half results are apparel/footwear or average score very low -> warn
    allowed_count = sum(1 for (r, s) in final if str(r.get("Category","")).strip() in allowed)
    avg_score = np.mean([s for (r, s) in final]) if final else 0.0
    if allowed_count < max(1, top_k//2) or avg_score < 0.22:
        st.error("We currently support search for clothing and footwear only. The uploaded image doesn't look like a clothing/footwear item. Try another image.")
        st.stop()

    # display results in grid
    st.markdown("---")
    st.subheader("Similar Products")
    ncols = min(5, len(final))
    cols = st.columns(ncols)
    for i, (row, sc) in enumerate(final):
        with cols[i % ncols]:
            candidate_src = row.get("ImageURL") or row.get("image_path") or row.get("Image")
            bts_c, ctype_c = fetch_image_bytes(candidate_src)
            if bts_c is None:
                st.markdown(f"<div class='card'><div class='placeholder'>Image not available</div><div class='meta'><b>{row.get('ProductTitle','')}</b><br/>Score: {sc:.3f}</div></div>", unsafe_allow_html=True)
            else:
                uri = bytes_to_data_uri(bts_c, ctype_c)
                st.markdown(f"<div class='card'><img src='{uri}' class='card-img' /><div class='meta'><b>{row.get('ProductTitle','')[:60]}</b><br/>Score: {sc:.3f}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; color:#475569; margin-top:10px;'>Tip: use Reset to start a new search.</div>", unsafe_allow_html=True)

else:
    st.markdown("<div style='text-align:center; color:#475569; margin-top:30px;'>Upload an image or paste a direct image URL in the sidebar, then click Search.</div>", unsafe_allow_html=True)
