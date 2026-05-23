"""
Multimodal Amazon Product RAG — Streamlit Chatbot
==================================================
Run with:
    streamlit run multimodal_rag_app.py

Prerequisites:
    pip install streamlit open-clip-torch torch torchvision
                chromadb Pillow requests huggingface_hub

The app connects to the ChromaDB vector store built in
multimodal_rag_amazon_products.ipynb and answers product
queries using text, image, or both modalities.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import time
import io
import json
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background: deep navy-to-slate ── */
.stApp {
    background: #080d14;
    min-height: 100vh;
}

/* Subtle grid texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,200,60,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,200,60,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.app-header {
    position: relative;
    padding: 2.8rem 2rem 2rem;
    text-align: center;
    border-bottom: 1px solid rgba(255,200,60,0.12);
    margin-bottom: 0;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(255,180,30,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.header-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff9f1c;
    margin-bottom: 0.7rem;
}
.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #f5ede0;
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.header-title em {
    font-style: italic;
    color: #ff9f1c;
}
.header-sub {
    color: #5a7080;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}
.amber-rule {
    width: 48px; height: 2px;
    background: linear-gradient(90deg, #ff9f1c, #ffcc44);
    margin: 1.1rem auto 0;
    border-radius: 2px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c1420 !important;
    border-right: 1px solid rgba(255,200,60,0.10);
}
.sb-section {
    margin-bottom: 1.5rem;
}
.sb-label {
    font-family: 'DM Mono', monospace;
    color: #ff9f1c;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    display: block;
}
.sb-divider {
    height: 1px;
    background: rgba(255,200,60,0.08);
    margin: 1.2rem 0;
}

/* ── Input overrides ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,200,60,0.18) !important;
    border-radius: 10px !important;
    color: #e8dfc8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    caret-color: #ff9f1c !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(255,159,28,0.55) !important;
    box-shadow: 0 0 0 3px rgba(255,159,28,0.08) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,200,60,0.22) !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e08a10 0%, #ff9f1c 50%, #ffc640 100%) !important;
    color: #0c0e12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.6rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 12px rgba(255,159,28,0.22) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(255,159,28,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Mode selector pills ── */
.mode-pills {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.mode-pill {
    flex: 1;
    text-align: center;
    padding: 0.45rem 0.5rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid rgba(255,200,60,0.18);
    background: rgba(255,255,255,0.02);
    color: #6a8090;
    transition: all 0.15s;
}
.mode-pill.active {
    background: rgba(255,159,28,0.14);
    border-color: rgba(255,159,28,0.5);
    color: #ff9f1c;
}

/* ── Strategy badge ── */
.strat-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255,159,28,0.10);
    border: 1px solid rgba(255,159,28,0.28);
    color: #ffb030;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
}

/* ── Chat bubble: user ── */
.bubble-user {
    display: flex;
    justify-content: flex-end;
    margin: 1.2rem 0 0.5rem;
    gap: 0.8rem;
    align-items: flex-start;
}
.bubble-user .bubble-body {
    max-width: 72%;
    background: rgba(255,159,28,0.12);
    border: 1px solid rgba(255,159,28,0.25);
    border-radius: 18px 4px 18px 18px;
    padding: 0.9rem 1.2rem;
    color: #f0e0c0;
    font-size: 1rem;
    line-height: 1.6;
}
.bubble-user .avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ff9f1c, #ffcc44);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
    color: #0c0e12;
    font-weight: 700;
}

/* ── Chat bubble: assistant ── */
.bubble-ai {
    display: flex;
    justify-content: flex-start;
    margin: 0.5rem 0 1.2rem;
    gap: 0.8rem;
    align-items: flex-start;
}
.bubble-ai .bubble-body {
    max-width: 85%;
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,200,60,0.14);
    border-left: 3px solid #ff9f1c;
    border-radius: 4px 18px 18px 18px;
    padding: 1.1rem 1.4rem;
    color: #e8dfc8;
    font-size: 0.97rem;
    line-height: 1.75;
}
.bubble-ai .avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: rgba(255,159,28,0.12);
    border: 1px solid rgba(255,159,28,0.28);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}

/* ── Image preview in chat ── */
.img-preview-wrap {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,200,60,0.14);
    border-radius: 10px;
    padding: 0.7rem;
    margin-top: 0.5rem;
    display: inline-block;
}
.img-preview-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #ff9f1c;
    margin-bottom: 0.4rem;
}

/* ── Product cards ── */
.products-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #5a7080;
    margin: 1.4rem 0 0.8rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,200,60,0.08);
}
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.product-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,200,60,0.12);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    transition: border-color 0.2s, background 0.2s;
}
.product-card:hover {
    border-color: rgba(255,159,28,0.32);
    background: rgba(255,159,28,0.05);
}
.product-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #ff9f1c;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
    opacity: 0.7;
}
.product-name {
    color: #d8cbb0;
    font-size: 0.88rem;
    font-weight: 500;
    line-height: 1.4;
    margin-bottom: 0.5rem;
}
.product-meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.product-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.04em;
}
.tag-price {
    background: rgba(100,200,100,0.10);
    color: #70d080;
    border: 1px solid rgba(100,200,100,0.2);
}
.tag-rating {
    background: rgba(255,200,60,0.08);
    color: #ffc030;
    border: 1px solid rgba(255,200,60,0.2);
}
.tag-category {
    background: rgba(100,160,220,0.08);
    color: #80b0d8;
    border: 1px solid rgba(100,160,220,0.15);
}
.tag-sim {
    background: rgba(180,100,220,0.08);
    color: #c090e0;
    border: 1px solid rgba(180,100,220,0.18);
}

/* ── Metrics row ── */
.metrics-strip {
    display: flex;
    gap: 0.7rem;
    margin: 0.6rem 0 1.2rem;
}
.metric-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,200,60,0.10);
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #ff9f1c;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a5060;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #283845;
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 1.2rem;
    opacity: 0.5;
}
.empty-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #344858;
    margin-bottom: 0.6rem;
}
.empty-sub {
    font-size: 0.88rem;
    color: #243040;
}

/* ── Selectbox / slider ── */
.stSelectbox label, .stSlider label, .stRadio label, label {
    color: #6a8090 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,200,60,0.18) !important;
    color: #e0d8c8 !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #ff9f1c !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    color: #6a8090 !important;
    border-color: rgba(255,200,60,0.08) !important;
    border-radius: 8px !important;
}

/* ── Spinner / alerts ── */
.stSpinner > div { border-top-color: #ff9f1c !important; }
.stAlert { border-radius: 10px !important; }

/* ── Dividers ── */
hr { border-color: rgba(255,200,60,0.08) !important; }

/* ── Caption ── */
.stCaption { color: #3a5060 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded backend (cached so CLIP/ChromaDB load only once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_clip(model_name="ViT-B-32", pretrained="openai"):
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device


@st.cache_resource(show_spinner=False)
def load_chroma(persist_dir: str, collection: str):
    import chromadb
    # Use a module-level embedding function (not a closure) so ChromaDB can
    # serialize it correctly — defining it inside a @cache_resource function
    # caused "could not pickle" connection errors.
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(name=collection, embedding_function=_CLIPEmb())
    return col


# ── CLIP encode helpers (module-level so cache fns can use them) ──────────────
# NOTE: load_clip() returns (model, preprocess, tokenizer, device) — 4 values.
# Always unpack all four; using _ as a placeholder for unused ones.

def _encode_texts(texts):
    import torch
    model, _preprocess, tokenizer, device = load_clip()  # unpack all 4 correctly
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        embs = model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy().astype("float32")


def _encode_image_pil(pil_img):
    import torch
    model, preprocess, _tokenizer, device = load_clip()
    tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype("float32")


def _fused_index_emb(text_emb, image_emb, alpha=0.5):
    """Reproduce the EXACT fusion used at index time in the notebook:
       fused = (1-alpha)*text + alpha*image, then L2-normalised.
       All ChromaDB vectors were stored this way — queries must match."""
    import numpy as np
    if image_emb is None:
        return text_emb
    if text_emb is None:
        # Image-only query: use a zero text component so fusion still works
        text_emb = np.zeros_like(image_emb)
    f = (1 - alpha) * text_emb + alpha * image_emb
    n = np.linalg.norm(f)
    return (f / n).astype("float32") if n > 0 else f


class _CLIPEmb:
    """Module-level ChromaDB EmbeddingFunction wrapping the CLIP text encoder."""

    def name(self) -> str:
        return "clip-vit-b32"

    def __call__(self, input):
        return _encode_texts(input).tolist()


def _fuse(t_emb, i_emb, alpha=0.5):
    import numpy as np
    if i_emb is None:
        return t_emb
    f = (1 - alpha) * t_emb + alpha * i_emb
    n = np.linalg.norm(f)
    return (f / n).astype("float32") if n > 0 else f


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query_text, query_image_pil, query_type, collection_obj, k, alpha):
    """Return list of result dicts from ChromaDB.
    
    IMPORTANT: ChromaDB vectors were indexed as fused(text+image) embeddings.
    All query modes must produce a fused vector in the same space to get
    meaningful cosine distances.
    """
    import numpy as np
    
    t_emb = _encode_texts([query_text])[0] if query_text and query_text.strip() else None
    i_emb = _encode_image_pil(query_image_pil) if query_image_pil is not None else None

    if query_type == "Text only":
        # At index time, text-only products used text_emb with alpha=0 image contribution
        # Match by querying with text emb fused against zero image vector
        emb = _fused_index_emb(t_emb, None, alpha=0.5)
    elif query_type == "Image only":
        # Fuse image against zero text to match indexed embedding space
        emb = _fused_index_emb(None, i_emb, alpha=0.5)
    else:  # Text + Image
        emb = _fused_index_emb(t_emb, i_emb, alpha=alpha)

    raw = collection_obj.query(
        query_embeddings=[emb.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for doc, meta, dist in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append({"doc": doc, "meta": meta, "distance": dist})
    return results


def unique_union(lists):
    seen, out = set(), []
    for lst in lists:
        for r in lst:
            key = r["doc"][:120]
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def rrf_merge(lists, k_rrf=60, top_n=6):
    scores, lookup = {}, {}
    for lst in lists:
        for rank, r in enumerate(lst):
            key = r["doc"][:120]
            lookup[key] = r
            scores[key] = scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [lookup[k] for k, _ in ranked[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# LLM via HF Inference API
# ─────────────────────────────────────────────────────────────────────────────

def llm_call(messages: list[dict], hf_token: str, model_id: str,
             max_tokens: int = 512, temperature: float = 0.1) -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Enter your Groq API key in the sidebar.")
    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"Groq API call failed: {e}\n"
            "Check that your API key is valid and the model name is correct."
        ) from e


RAG_SYSTEM = """You are a helpful Amazon product shopping assistant.
Use ONLY the product information provided in the context below.
For each product you recommend:
  - State the exact product name as it appears in the context.
  - Quote the exact price and rating from the context (do not estimate).
  - Explain in 1–2 sentences why it matches the query.
If a product in the context closely matches but is not a perfect match, recommend it and note the difference.
You MUST recommend products from the context. If retrieved products are present, always recommend the top match by name, price, and rating — never say you couldn't find a match when products are listed in the context.
Only say "I couldn't find a matching product" if the context is completely empty with zero products listed.
Never invent product names, prices, ratings, or features not present in the context."""


def format_context(results):
    parts = []
    for i, r in enumerate(results, 1):
        meta = r["meta"]
        line = f"[Product {i}]\n{r['doc']}"
        img = meta.get("image_url", "")
        if img:
            line += f"\nImage URL: {img}"
        parts.append(line)
    return "\n\n---\n\n".join(parts)

def _describe_image_for_llm(pil_img) -> str:
    """
    Generate a concise text description of the uploaded image using CLIP
    zero-shot classification against common product categories.
    This bridges the gap between image retrieval and LLM text reasoning.
    """
    candidate_labels = [
    "gaming headset", "wireless headphones", "electronics", "bluetooth speaker",
    "laptop accessory", "keyboard", "mouse", "monitor", "gaming controller",
    "sports merchandise", "clothing", "kitchen appliance", "toy", "book",
    "home decor", "beauty product", "office supply", "collectible",
    ]
    text_embs = _encode_texts(candidate_labels)
    img_emb = _encode_image_pil(pil_img)
    sims = text_embs @ img_emb  # cosine similarity (all L2-normed)
    top_idx = sims.argsort()[-3:][::-1]
    top_labels = [candidate_labels[i] for i in top_idx]
    return f"The uploaded image appears to show: {', '.join(top_labels)}"


def _effective_query(query, query_type, query_image_pil=None):
    if query.strip():
        return query.strip()
    if query_type == "Image only" and query_image_pil is not None:
        desc = _describe_image_for_llm(query_image_pil)
        return (
            f"{desc}. "
            "Based on the retrieved products below, identify the closest match. "
            "State its exact name, price, and explain why it visually matches the uploaded image."
        )
    return query



def answer_standard(query, results, hf_token, model_id, query_type="Text only", query_image_pil=None):
    ctx = format_context(results)
    q = _effective_query(query, query_type, query_image_pil)
    return llm_call([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ], hf_token, model_id)


def answer_multi_query(query, collection_obj, k, alpha, query_image_pil,
                       query_type, hf_token, model_id, n_alt=3):
    q = _effective_query(query, query_type, query_image_pil)
    # For image-only mode skip LLM alt-query generation (no text to rephrase);
    # use the image retrieval directly with a single pass
    if query_type == "Image only":
        results = retrieve(query, query_image_pil, query_type, collection_obj, k, alpha)
        alternatives = []
    else:
        raw_alts = llm_call([
            {"role": "system", "content": "Output only alternative search queries, one per line, no numbering."},
            {"role": "user",   "content": f"Generate {n_alt} alternative Amazon product search phrasings for:\n{q}"},
        ], hf_token, model_id, max_tokens=200)
        alternatives = [a.strip() for a in raw_alts.splitlines() if a.strip()][:n_alt]
        all_queries = [q] + alternatives
        lists = [retrieve(aq, query_image_pil, "Text only", collection_obj, k, alpha)
                 for aq in all_queries]
        results = unique_union(lists)
    ctx = format_context(results)
    answer = llm_call([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ], hf_token, model_id)
    return answer, results, alternatives


def answer_rag_fusion(query, collection_obj, k, alpha, query_image_pil,
                      query_type, hf_token, model_id, n_alt=3):
    q = _effective_query(query, query_type, query_image_pil)
    if query_type == "Image only":
        results = retrieve(query, query_image_pil, query_type, collection_obj, k, alpha)
        alternatives = []
    else:
        raw_alts = llm_call([
            {"role": "system", "content": "Output only alternative search queries, one per line, no numbering."},
            {"role": "user",   "content": f"Generate {n_alt} alternative Amazon product search phrasings for:\n{q}"},
        ], hf_token, model_id, max_tokens=200)
        alternatives = [a.strip() for a in raw_alts.splitlines() if a.strip()][:n_alt]
        all_queries = [q] + alternatives
        lists = [retrieve(aq, query_image_pil, "Text only", collection_obj, k, alpha)
                 for aq in all_queries]
        results = rrf_merge(lists, top_n=k)
    ctx = format_context(results)
    answer = llm_call([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ], hf_token, model_id)
    return answer, results, alternatives


def answer_hyde(query, collection_obj, k, alpha, query_image_pil,
                query_type, hf_token, model_id):
    q = _effective_query(query, query_type, query_image_pil)
    if query_type == "Image only":
        # HyDE needs text to generate a hypothesis — fall back to standard image retrieval
        results = retrieve(query, query_image_pil, query_type, collection_obj, k, alpha)
        hypo = "(image-only query — HyDE skipped hypothesis generation)"
    else:
        hypo = llm_call([
            {"role": "system", "content": "Write a realistic Amazon product listing (name, features, price) that answers the query."},
            {"role": "user",   "content": f"Query: {q}\n\nHypothetical product listing:"},
        ], hf_token, model_id, max_tokens=200)
        hypo_short = " ".join(hypo.split()[:60])
        hypo_emb = _encode_texts([hypo_short])[0]
        col_raw = collection_obj.query(
            query_embeddings=[hypo_emb.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = [
            {"doc": d, "meta": m, "distance": dist}
            for d, m, dist in zip(
                col_raw["documents"][0], col_raw["metadatas"][0], col_raw["distances"][0]
            )
        ]
    ctx = format_context(results)
    answer = llm_call([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ], hf_token, model_id)
    return answer, results, hypo


def answer_step_back(query, collection_obj, k, alpha, query_image_pil,
                     query_type, hf_token, model_id):
    q = _effective_query(query, query_type, query_image_pil)
    if query_type == "Image only":
        # No text to abstract — just do image retrieval
        results = retrieve(query, query_image_pil, query_type, collection_obj, k, alpha)
        abstract = "(image-only query — step-back abstraction skipped)"
    else:
        abstract = llm_call([
            {"role": "user", "content":
             f"Rewrite this specific product question as a broader product category question.\n"
             f"Specific: {q}\nAbstract:"},
        ], hf_token, model_id, max_tokens=80)
        abstract = abstract.strip().splitlines()[0]
        spec = retrieve(query, query_image_pil, query_type, collection_obj, k, alpha)
        abst = retrieve(abstract, None, "Text only", collection_obj, k, 0.5)
        results = unique_union([spec, abst])

    ctx = format_context(results)
    answer = llm_call([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ], hf_token, model_id)
    return answer, results, abstract


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


STRATEGY_INFO = {
    "Standard":     "Direct CLIP similarity search — fastest, reliable for clear queries.",
    "Multi-Query":  "LLM generates alternative phrasings; results are unioned — great for ambiguous questions.",
    "RAG-Fusion":   "Alternative phrasings reranked by Reciprocal Rank Fusion — highest retrieval quality.",
    "HyDE":         "Generates a hypothetical product listing first, then searches — bridges casual language and catalogue terms.",
    "Step-Back":    "Retrieves for both the specific and a broader category query — useful for 'what type of X…' questions.",
}

SAMPLE_QUERIES = [
    "Wireless headphones with good noise cancellation under $80",
    "Portable Bluetooth speaker for outdoor use, waterproof",
    "Laptop stand for working from home, adjustable height",
    "USB-C hub with HDMI and multiple USB ports",
    "Gift ideas for someone who loves cooking",
    "Running shoes with extra cushioning for long distances",
    "Mechanical keyboard compact tenkeyless",
    "Ring light for video calls and streaming",
]


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("total_queries", 0),
    ("total_products", 0),
    ("query_input", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-eyebrow">🛍️ Powered by CLIP · Groq · ChromaDB</div>
  <h1 class="header-title">Amazon Product <em>AI</em></h1>
  <p class="header-sub">Search by text, image, or both — multimodal product intelligence</p>
  <div class="amber-rule"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    # ── Credentials ──
    st.markdown('<span class="sb-label">🔑 Groq API Key</span>', unsafe_allow_html=True)
    groq_key_input = st.text_input(
        "Groq Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        label_visibility="collapsed",
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
    hf_token = ""  # kept for function signatures, unused

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Vector store ──
    st.markdown('<span class="sb-label">📁 ChromaDB Path</span>', unsafe_allow_html=True)
    persist_dir = st.text_input(
        "DB Path", value="./chroma_amazon_mm",
        label_visibility="collapsed",
    )

    st.markdown('<span class="sb-label">📦 Collection Name</span>', unsafe_allow_html=True)
    collection_name = st.text_input(
        "Collection", value="amazon_multimodal",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Model ──
    st.markdown('<span class="sb-label">🤖 LLM Model</span>', unsafe_allow_html=True)
    model_id = st.selectbox(
        "Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        label_visibility="collapsed",
    )

    # ── Retrieval params ──
    st.markdown('<span class="sb-label">🔍 Products to Retrieve (k)</span>', unsafe_allow_html=True)
    k_val = st.slider("k", min_value=2, max_value=12, value=6, label_visibility="collapsed")

    st.markdown('<span class="sb-label">⚖️ Image/Text Fusion Weight (α)</span>', unsafe_allow_html=True)
    alpha = st.slider(
        "alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = text only, 0.5 = equal, 1.0 = image only",
        label_visibility="collapsed",
    )
    st.caption(f"Text weight: {1-alpha:.0%}   Image weight: {alpha:.0%}")

    # ── Strategy ──
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">⚡ Query Strategy</span>', unsafe_allow_html=True)
    strategy = st.selectbox(
        "Strategy", options=list(STRATEGY_INFO.keys()),
        index=0, label_visibility="collapsed",
    )
    st.caption(STRATEGY_INFO[strategy])

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">🗄️ Database Status</span>', unsafe_allow_html=True)

    if Path(persist_dir).exists():
        try:
            import chromadb
            _client = chromadb.PersistentClient(path=persist_dir)
            _col = _client.get_collection(name=collection_name)
            _count = _col.count()
            _db_mtime = os.path.getmtime(persist_dir)
            _db_age = time.strftime("%Y-%m-%d %H:%M", time.localtime(_db_mtime))
            st.success(f"✅ Connected: **{_count:,}** products")
            st.caption(f"📅 DB last modified: {_db_age}")
            st.caption(f"📁 `{persist_dir}`")
        except Exception as _e:
            st.error(f"⚠️ Could not read DB: {_e}")
    else:
        st.error(f"⚠️ Path not found: `{persist_dir}`")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)


    # ── Session stats ──
    st.markdown('<span class="sb-label">📊 Session Stats</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("Products", st.session_state.total_products)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_queries = 0
        st.session_state.total_products = 0
        st.rerun()

    # ── History ──
    if st.session_state.history:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-label">🕑 Recent</span>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-6:]):
            short = item["query"][:55] + ("…" if len(item["query"]) > 55 else "")
            mode_icon = {"Text only": "💬", "Image only": "🖼️", "Text + Image": "🔀"}.get(
                item.get("query_type", "Text only"), "💬"
            )
            st.caption(f"{mode_icon} {short}")
            st.caption(f"_↳ {item['strategy']} · {item['elapsed']:.1f}s · {len(item['results'])} products_")


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
col_chat, col_gap = st.columns([3, 1])

with col_chat:

    # ── Query type selector ──
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    query_type = st.radio(
        "Query mode",
        options=["Text only", "Image only", "Text + Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Text input ──
    query_disabled = (query_type == "Image only")
    query = st.text_area(
        "Your question",
        placeholder="e.g. Wireless headphones with noise cancellation under $60",
        height=90,
        label_visibility="collapsed",
        disabled=query_disabled,
        value="" if query_disabled else st.session_state.get("query_input", ""),
    )

    # ── Image uploader ──
    uploaded_image = None
    query_image_pil = None
    if query_type in ("Image only", "Text + Image"):
        uploaded_image = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_image:
            from PIL import Image
            uploaded_image.seek(0)                        # reset buffer position
            query_image_pil = Image.open(uploaded_image).convert("RGB")
            query_image_pil.load()                        # force full decode before stream closes

    # ── Submit row ──
    btn_col, badge_col = st.columns([1, 3])
    with btn_col:
        submitted = st.button("Search →", use_container_width=True)
    with badge_col:
        st.markdown(
            f'<div style="padding-top:0.6rem">'
            f'<span class="strat-badge">⚡ {strategy}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Sample queries ──
    with st.expander("💡 Sample queries to try"):
        cols = st.columns(2)
        for i, s in enumerate(SAMPLE_QUERIES):
            if cols[i % 2].button(s, key=f"sample_{i}"):
                query = s
                submitted = True

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # Run query
    # ─────────────────────────────────────────────────────────────────
    if submitted:
        text_ok  = bool(query.strip()) or query_type == "Image only"
        image_ok = (query_image_pil is not None) or (query_type == "Text only")

        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ Enter your Groq API key in the sidebar.")
        elif not Path(persist_dir).exists():
            st.error(f"⚠️ ChromaDB not found at `{persist_dir}`. Run the indexing notebook first.")
        elif not text_ok:
            st.warning("Please enter a question.")
        elif query_type in ("Image only", "Text + Image") and query_image_pil is None:
            st.warning("Please upload an image.")
        else:
            display_query = query.strip() if query.strip() else "(image query)"
            with st.spinner("🔍 Searching catalogue…"):
                try:
                    t0 = time.time()

                    # Load resources
                    with st.spinner("Loading CLIP encoder…"):
                        load_clip()
                    with st.spinner("Connecting to ChromaDB…"):
                        col_obj = load_chroma(persist_dir, collection_name)

                    extra_info = {}

                    if strategy == "Standard":
                        results = retrieve(
                            query.strip(), query_image_pil, query_type,
                            col_obj, k_val, alpha
                        )
                        answer = answer_standard(query, results, hf_token, model_id, query_type, query_image_pil)

                    elif strategy == "Multi-Query":
                        answer, results, alts = answer_multi_query(
                            display_query, col_obj, k_val, alpha,
                            query_image_pil, query_type, hf_token, model_id
                        )
                        extra_info["alternatives"] = alts

                    elif strategy == "RAG-Fusion":
                        answer, results, alts = answer_rag_fusion(
                            display_query, col_obj, k_val, alpha,
                            query_image_pil, query_type, hf_token, model_id
                        )
                        extra_info["alternatives"] = alts

                    elif strategy == "HyDE":
                        answer, results, hypo = answer_hyde(
                            display_query, col_obj, k_val, alpha,
                            query_image_pil, query_type, hf_token, model_id
                        )
                        extra_info["hypothetical"] = hypo

                    elif strategy == "Step-Back":
                        answer, results, abstract = answer_step_back(
                            display_query, col_obj, k_val, alpha,
                            query_image_pil, query_type, hf_token, model_id
                        )
                        extra_info["abstract_query"] = abstract

                    elapsed = time.time() - t0

                    st.session_state.history.append({
                        "query":      display_query,
                        "answer":     answer,
                        "results":    results,
                        "strategy":   strategy,
                        "query_type": query_type,
                        "image_b64":  pil_to_b64(query_image_pil) if query_image_pil else None,
                        "elapsed":    elapsed,
                        "extra":      extra_info,
                    })
                    st.session_state.total_queries += 1
                    st.session_state.total_products += len(results)

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    # ─────────────────────────────────────────────────────────────────
    # Render chat history (newest first in display, newest at bottom)
    # ─────────────────────────────────────────────────────────────────
    if st.session_state.history:

        # Latest exchange
        latest = st.session_state.history[-1]

        # ── Metrics strip ──
        n_words = len(latest["answer"].split())
        sim_pct = round((1 - latest["results"][0]["distance"]) * 100) if latest["results"] else 0
        st.markdown(
            f'<div class="metrics-strip">'
            f'  <div class="metric-chip"><div class="metric-val">{latest["elapsed"]:.1f}s</div>'
            f'    <div class="metric-lbl">Response time</div></div>'
            f'  <div class="metric-chip"><div class="metric-val">{len(latest["results"])}</div>'
            f'    <div class="metric-lbl">Products found</div></div>'
            f'  <div class="metric-chip"><div class="metric-val">{n_words}</div>'
            f'    <div class="metric-lbl">Answer words</div></div>'
            f'  <div class="metric-chip"><div class="metric-val">{sim_pct}%</div>'
            f'    <div class="metric-lbl">Top similarity</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── User bubble ──
        mode_icon = {"Text only": "💬", "Image only": "🖼️", "Text + Image": "🔀"}.get(
            latest.get("query_type", "Text only"), "💬"
        )
        img_html = ""
        if latest.get("image_b64"):
            img_html = (
                f'<div class="img-preview-wrap">'
                f'  <div class="img-preview-label">📎 Query image</div>'
                f'  <img src="data:image/jpeg;base64,{latest["image_b64"]}" '
                f'       style="max-width:220px;max-height:180px;border-radius:6px;display:block;">'
                f'</div>'
            )

        st.markdown(
            f'<div class="bubble-user">'
            f'  <div class="bubble-body">'
            f'    {latest["query"]}'
            f'    {img_html}'
            f'  </div>'
            f'  <div class="avatar">U</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── AI bubble ──
        st.markdown(
            f'<div class="bubble-ai">'
            f'  <div class="avatar">🛍</div>'
            f'  <div class="bubble-body">{latest["answer"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Extra strategy info ──
        extra = latest.get("extra", {})
        if extra.get("alternatives"):
            with st.expander("🔀 Alternative queries generated"):
                for alt in extra["alternatives"]:
                    st.markdown(f"- `{alt}`")
        if extra.get("hypothetical"):
            with st.expander("💭 Hypothetical product used for HyDE search"):
                st.markdown(f"_{extra['hypothetical'][:400]}_")
        if extra.get("abstract_query"):
            with st.expander("🔭 Step-Back abstract query"):
                st.markdown(f"_{extra['abstract_query']}_")

        # ── Retrieved product cards ──
        if latest["results"]:
            st.markdown(
                f'<div class="products-header">📦 {len(latest["results"])} retrieved products</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="product-grid">', unsafe_allow_html=True)
            cards_html = ""
            for i, r in enumerate(latest["results"], 1):
                meta = r["meta"]
                name = meta.get("product_name", "Unknown product")[:90]
                cat  = meta.get("category", "")[:40]
                price = meta.get("price", "")
                rating = meta.get("rating", "")
                sim = round((1 - r["distance"]) * 100)
                img_url = meta.get("image_url", "")

                tags = ""
                if price:   tags += f'<span class="product-tag tag-price">💲 {price}</span>'
                if rating:  tags += f'<span class="product-tag tag-rating">⭐ {rating}</span>'
                if cat:     tags += f'<span class="product-tag tag-category">📂 {cat}</span>'
                tags += f'<span class="product-tag tag-sim">~{sim}% match</span>'

                thumb = ""
                if img_url:
                    thumb = (
                        f'<img src="{img_url}" '
                        f'style="width:100%;max-height:100px;object-fit:contain;'
                        f'border-radius:6px;margin-bottom:0.5rem;background:#111;" '
                        f'onerror="this.style.display=\'none\'">'
                    )

                cards_html += (
                    f'<div class="product-card">'
                    f'  <div class="product-rank">#{i}</div>'
                    f'  {thumb}'
                    f'  <div class="product-name">{name}</div>'
                    f'  <div class="product-meta">{tags}</div>'
                    f'</div>'
                )
            st.markdown(cards_html + '</div>', unsafe_allow_html=True)

        # ── Previous exchanges ──
        if len(st.session_state.history) > 1:
            st.markdown("---")
            with st.expander(f"📚 Chat history ({len(st.session_state.history) - 1} earlier)"):
                for item in reversed(st.session_state.history[:-1]):
                    qtype_icon = {"Text only": "💬", "Image only": "🖼️", "Text + Image": "🔀"}.get(
                        item.get("query_type", "Text only"), "💬"
                    )
                    st.markdown(
                        f'<div style="color:#6a8090;font-size:0.82rem;font-style:italic;'
                        f'margin:0.9rem 0 0.25rem;">'
                        f'{qtype_icon} &nbsp;"{item["query"]}"</div>',
                        unsafe_allow_html=True,
                    )
                    short_ans = item["answer"][:420] + ("…" if len(item["answer"]) > 420 else "")
                    st.markdown(
                        f'<div style="color:#c0b090;font-size:0.9rem;line-height:1.65;'
                        f'margin-bottom:0.4rem;">{short_ans}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"⚡ {item['strategy']}  ·  ⏱ {item['elapsed']:.1f}s  "
                        f"·  📦 {len(item['results'])} products"
                    )
                    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    else:
        # ── Empty state ──
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🛍️</div>
          <div class="empty-title">Ready to find products</div>
          <div class="empty-sub">
            Type a question, upload an image, or try both together.<br>
            Switch query strategies in the sidebar to compare results.
          </div>
        </div>
        """, unsafe_allow_html=True)
