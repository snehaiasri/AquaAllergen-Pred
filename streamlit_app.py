import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from huggingface_hub import snapshot_download

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

import base64

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

import base64
import streamlit as st

def set_home_background_video(video_path: str):
    """Set a looping MP4 background video only on the Home page."""
    path = Path(video_path)
    if not path.exists():
        st.warning(f"Home background video not found: {video_path}")
        return

    encoded = file_to_base64(path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        #home-bg-video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -100;
            object-fit: cover;
            filter: brightness(0.72);
        }}
        .home-hero {{
            background: rgba(255, 255, 255, 0.84);
            border-radius: 22px;
            padding: 2.2rem 2.5rem;
            margin-top: 2rem;
            box-shadow: 0 18px 45px rgba(0,0,0,0.18);
            backdrop-filter: blur(4px);
            max-width: 980px;
        }}
        .home-hero h1 {{
            font-size: 3.0rem;
            line-height: 1.08;
            margin-bottom: 0.4rem;
            color: #102033;
        }}
        .home-hero .subtitle {{
            font-size: 1.15rem;
            color: #34495e;
            margin-bottom: 1.3rem;
        }}
        .home-card-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.2rem;
        }}
        .home-mini-card {{
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(16,32,51,0.12);
            border-radius: 16px;
            padding: 1rem;
        }}
        .home-note {{
            margin-top: 1.2rem;
            padding: 1rem;
            border-radius: 14px;
            background: rgba(230, 244, 255, 0.85);
            border-left: 5px solid #2b7bbb;
            color: #17324d;
        }}
        </style>
        <video autoplay muted loop playsinline id="home-bg-video">
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="AquaAllergen-Pred",
    page_icon="🧬",
    layout="wide",
)

@st.cache_resource(show_spinner=True)
def download_hf_assets():
    local_dir = snapshot_download(
        repo_id="smurmu/AquaAllergen-Pred-assets",
        repo_type="dataset",
        local_dir="./hf_assets",
        local_dir_use_symlinks=False
    )
    return Path(local_dir)

ASSET_DIR = download_hf_assets()

APP_TITLE = "AquaAllergen-Pred"
APP_SUBTITLE = "LLM-based protein sequence allergen prediction using a machine learning classifier for aquatic species"

DEFAULT_MODEL_DIR = str(ASSET_DIR / "esm2_8M_model")
DEFAULT_CLASSIFIER = "./fish_model_logreg_fold1.joblib"
DEFAULT_PREDICTED_ROOT = str(ASSET_DIR / "predicted_allergens")
DEFAULT_IMAGE_DIR = str(ASSET_DIR / "images")
DEFAULT_HOME_VIDEO = str(ASSET_DIR / "images" / "loop_video3.mp4")

CATEGORY_CONFIG = {
    "Fish": {
        "folder": "fish",
        "emoji": "",
        "image": "fish_image.jpg",
        "description": "Predicted allergen-like proteins in fish species.",
    },
    "Crustaceans": {
        "folder": "crustaceans",
        "emoji": "",
        "image": "crustaceans_image.jpg",
        "description": "Predicted allergen-like proteins in crustaceans species.",
    },
    "Molluscs": {
        "folder": "molluscs",
        "emoji": "",
        "image": "molluscs_image.jpg",
        "description": "Predicted allergen-like proteins in molluscan species.",
    },
}

MAX_LEN = 1022
BATCH_SIZE = 8
PRED_THRESHOLD = 0.5

# -----------------------------
# HELPERS: sequence prediction
# -----------------------------
def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]", "X", seq)
    return seq


def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))


def parse_fasta_text(text: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_parts = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, clean_sequence("".join(seq_parts))))
            header = line[1:].strip() or f"Sequence_{len(records)+1}"
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, clean_sequence("".join(seq_parts))))
    return [(h.split()[0], s) for h, s in records if s]


def parse_uploaded_fasta(uploaded_file) -> List[Tuple[str, str]]:
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return parse_fasta_text(text)


@st.cache_resource(show_spinner=False)
def load_models(model_dir: str, classifier_path: str):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    esm_model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    esm_model.eval()
    clf = joblib.load(classifier_path)
    return tokenizer, esm_model, clf, device


@torch.no_grad()
def embed_batch(tokenizer, model, seqs: List[str], device: str) -> np.ndarray:
    tok = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok)
    h = out.last_hidden_state
    mask = tok.get("attention_mask", None)
    if mask is None:
        emb = h.mean(dim=1)
    else:
        mask = mask.unsqueeze(-1).type_as(h)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        emb = summed / denom
    return emb.detach().cpu().numpy().astype(np.float32)


def predict_sequences(records: List[Tuple[str, str]], tokenizer, esm_model, clf, device: str) -> Tuple[pd.DataFrame, np.ndarray]:
    ids, seqs, lengths = [], [], []
    for rid, seq in records:
        if seq:
            ids.append(rid)
            seqs.append(seq[:MAX_LEN])
            lengths.append(len(seq))

    if not seqs:
        return pd.DataFrame(), np.empty((0, 0))

    X_parts = []
    for i in range(0, len(seqs), BATCH_SIZE):
        X_parts.append(embed_batch(tokenizer, esm_model, seqs[i:i+BATCH_SIZE], device))
    X = np.vstack(X_parts)

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= PRED_THRESHOLD).astype(int)
    labels = np.where(preds == 1, "Allergen-like", "Non-allergen-like")

    df = pd.DataFrame({
        "Protein_ID": ids,
        "Protein_Sequence": seqs,
        "Length": lengths,
        "Prediction_Probability": probs,
        "Predicted_Class": labels,
    })
    return df, X


def get_feature_direction(clf, embedding: np.ndarray, top_n: int = 10) -> str:
    """Simple global explanation from logistic regression coefficients."""
    if not hasattr(clf, "coef_"):
        return "Global coefficient-based explanation is unavailable for this classifier."

    coefs = clf.coef_[0]
    contributions = embedding * coefs
    idx = np.argsort(np.abs(contributions))[::-1][:top_n]

    lines = []
    for i in idx:
        direction = "toward allergen" if contributions[i] > 0 else "toward non-allergen"
        lines.append(f"Embedding feature {i}: contribution {contributions[i]:.4f} ({direction})")
    return "\n".join(lines)


def segment_occlusion_explanation(seq: str, tokenizer, esm_model, clf, device: str, n_segments: int = 8) -> pd.DataFrame:
    """
    Lightweight explainability by segment occlusion.
    Removes one segment at a time and measures probability drop.
    Positive delta means that segment supported the allergen prediction.
    """
    seq = clean_sequence(seq)
    if len(seq) < 20:
        n_segments = min(n_segments, 4)

    original_emb = embed_batch(tokenizer, esm_model, [seq], device)[0]
    original_prob = float(clf.predict_proba(original_emb.reshape(1, -1))[:, 1][0])

    segments = []
    seg_size = max(1, len(seq) // n_segments)

    start = 0
    while start < len(seq):
        end = min(len(seq), start + seg_size)
        occluded_seq = seq[:start] + seq[end:]
        if len(occluded_seq) == 0:
            occluded_seq = "X"
        occ_emb = embed_batch(tokenizer, esm_model, [occluded_seq], device)[0]
        occ_prob = float(clf.predict_proba(occ_emb.reshape(1, -1))[:, 1][0])
        delta = original_prob - occ_prob
        segments.append({
            "Segment": f"{start+1}-{end}",
            "Sequence": seq[start:end],
            "Original_Prob": original_prob,
            "Occluded_Prob": occ_prob,
            "Probability_Drop": delta,
            "Interpretation": (
                "Supports allergen prediction" if delta > 0 else "Supports non-allergen prediction / negligible"
            )
        })
        start = end

    return pd.DataFrame(segments).sort_values("Probability_Drop", ascending=False).reset_index(drop=True)


def color_class(prob: float) -> str:
    if prob >= 0.9:
        return "🟥 High-confidence allergen-like"
    if prob >= 0.7:
        return "🟧 Moderate allergen-like"
    if prob >= 0.5:
        return "🟨 Borderline allergen-like"
    return "🟦 Non-allergen-like"

# -----------------------------
# HELPERS: predicted allergen database tab
# -----------------------------
def clean_species_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_predictions$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ").strip()
    return name


def find_prediction_files(predicted_root: str, category_folder: str) -> Dict[str, Path]:
    """
    Finds species-wise CSV files. Expected structure is predicted_root/category_folder/*.csv.
    If the category folder does not exist, it returns an empty dictionary.
    """
    folder = Path(predicted_root) / category_folder
    if not folder.exists():
        return {}

    files = sorted(folder.glob("*.csv"))
    return {clean_species_name(p): p for p in files}


def normalize_prediction_df(df: pd.DataFrame, species_name: Optional[str] = None) -> pd.DataFrame:
    """Standardizes column names from different prediction CSV versions."""
    rename_map = {}
    candidates = {
        "Protein_ID": ["Protein_ID", "UniProtID", "UniProt_ID", "Accession", "protein_id"],
        "Protein_Sequence": ["Protein_Sequence", "Protein_Seq", "Sequence", "sequence"],
        "Protein_Length": ["Protein_Length", "Length", "length"],
        "Prediction_Probability": ["Prediction_Probability", "Prediction_Score", "Probability", "probability", "Score"],
        "Predicted_Class": ["Predicted_Class", "Class_Label", "Prediction", "predicted_class"],
        "Species": ["Species", "species"],
    }
    for standard, cols in candidates.items():
        for c in cols:
            if c in df.columns:
                rename_map[c] = standard
                break
    df = df.rename(columns=rename_map).copy()

    if "Species" not in df.columns:
        df.insert(0, "Species", species_name if species_name else "Unknown")

    if "Protein_Length" not in df.columns and "Protein_Sequence" in df.columns:
        df["Protein_Length"] = df["Protein_Sequence"].astype(str).str.len()

    if "Prediction_Probability" in df.columns:
        df["Prediction_Probability"] = pd.to_numeric(df["Prediction_Probability"], errors="coerce")

    if "Predicted_Class" in df.columns:
        # Normalize common labels
        df["Predicted_Class"] = df["Predicted_Class"].astype(str).replace({
            "1": "Allergen",
            "0": "Non-allergen",
            "Allergen-like": "Allergen",
            "Non-allergen-like": "Non-allergen",
        })

    preferred_order = [
        "Species", "Protein_ID", "Protein_Sequence", "Protein_Length",
        "Prediction_Probability", "Predicted_Class"
    ]
    other_cols = [c for c in df.columns if c not in preferred_order]
    available = [c for c in preferred_order if c in df.columns] + other_cols
    return df[available]


@st.cache_data(show_spinner=False)
def load_prediction_csv_cached(file_path_str: str, species_name: str, file_mtime: float) -> pd.DataFrame:
    """Cached CSV loader. file_mtime is included so cache refreshes when file changes."""
    possible_cols = [
        "Species", "species",
        "Protein_ID", "UniProtID", "UniProt_ID", "Accession", "protein_id",
        "Protein_Sequence", "Protein_Seq", "Sequence", "sequence",
        "Protein_Length", "Length", "length",
        "Prediction_Probability", "Prediction_Score", "Probability", "probability", "Score",
        "Predicted_Class", "Class_Label", "Prediction", "predicted_class",
    ]
    try:
        df = pd.read_csv(file_path_str, usecols=lambda c: c in set(possible_cols))
    except Exception:
        df = pd.read_csv(file_path_str)
    return normalize_prediction_df(df, species_name=species_name)


def load_prediction_csv(file_path: Path, species_name: str) -> pd.DataFrame:
    return load_prediction_csv_cached(str(file_path), species_name, file_path.stat().st_mtime)


def load_multiple_prediction_csvs(files: Dict[str, Path]) -> pd.DataFrame:
    dfs = []
    for species, path in files.items():
        try:
            dfs.append(load_prediction_csv(path, species))
        except Exception as e:
            st.warning(f"Could not read {path.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def filter_prediction_table(df: pd.DataFrame, class_filter: str, min_probability: float) -> pd.DataFrame:
    filtered = df.copy()
    if "Prediction_Probability" in filtered.columns:
        filtered = filtered[filtered["Prediction_Probability"] >= min_probability]
    if class_filter != "All classes" and "Predicted_Class" in filtered.columns:
        if class_filter == "Predicted allergens only":
            filtered = filtered[filtered["Predicted_Class"].str.contains("Allergen", case=False, na=False)]
            filtered = filtered[~filtered["Predicted_Class"].str.contains("Non", case=False, na=False)]
        elif class_filter == "Predicted non-allergens only":
            filtered = filtered[filtered["Predicted_Class"].str.contains("Non", case=False, na=False)]
    if "Prediction_Probability" in filtered.columns:
        filtered = filtered.sort_values("Prediction_Probability", ascending=False)
    return filtered.reset_index(drop=True)



def resolve_category_image(image_dir: str, filename: str) -> Optional[Path]:
    """Find a real image for the category card.
    Put real images in images folder, e.g. fish.jpg, crustaceans.jpg, molluscs.jpg.
    """
    base = Path(image_dir)
    stem = Path(filename).stem
    candidates = [base / filename]
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidates.append(base / f"{stem}{ext}")
    for c in candidates:
        if c.exists():
            return c
    return None


def category_card(label: str, cfg: Dict, selected: bool, image_dir: str) -> bool:
    img_path = resolve_category_image(image_dir, cfg.get("image", ""))

    with st.container(border=True):

        # --- FIXED SIZE IMAGE ---
        if img_path is not None:
            st.markdown(
                f"""
                <div style="
                    height: 220px;
                    overflow: hidden;
                    border-radius: 10px;
                ">
                    <img src="data:image/jpg;base64,{img_to_base64(img_path)}"
                         style="width:100%; height:100%; object-fit:cover;">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning(f"Image not found: {cfg.get('image', '')}")

        clicked = st.button(
            label,
            use_container_width=True,
            type=("primary" if selected else "secondary"),
            key=f"cat_{label}"
        )

        st.caption(cfg.get("description", ""))

    return clicked


def pagination_controls(total_pages: int, rows_per_page: int) -> None:
    nav1, nav2, nav3, nav4, nav5 = st.columns([0.8, 0.8, 2.2, 0.8, 0.8])
    with nav1:
        if st.button("⏮ First", disabled=st.session_state.pred_page <= 1, key="pg_first_bottom"):
            st.session_state.pred_page = 1
            st.rerun()
    with nav2:
        if st.button("<", disabled=st.session_state.pred_page <= 1, key="pg_prev_bottom"):
            st.session_state.pred_page -= 1
            st.rerun()
    with nav3:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.45rem;'>"
            f"Page <b>{st.session_state.pred_page}</b> of <b>{total_pages}</b> "
            f"({rows_per_page} records/page)</div>",
            unsafe_allow_html=True,
        )
    with nav4:
        if st.button(">", disabled=st.session_state.pred_page >= total_pages, key="pg_next_bottom"):
            st.session_state.pred_page += 1
            st.rerun()
    with nav5:
        if st.button("Last ⏭", disabled=st.session_state.pred_page >= total_pages, key="pg_last_bottom"):
            st.session_state.pred_page = total_pages
            st.rerun()

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Prediction", "Predicted Allergens", "Tutorial", "Contact"])

# Internal paths/settings are kept hidden from users.
# Edit these constants above if the local model, classifier, image, video,
# or predicted-result folders are moved.
model_dir = DEFAULT_MODEL_DIR
classifier_path = DEFAULT_CLASSIFIER
predicted_root = DEFAULT_PREDICTED_ROOT
image_dir = DEFAULT_IMAGE_DIR
home_video_path = DEFAULT_HOME_VIDEO
show_debug = False

if page != "Home":
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

if page == "Home":
    set_home_background_video(home_video_path)
    st.markdown(
        f"""
        <div class="home-hero">
            <h1>{APP_TITLE}</h1>
            <div class="subtitle">{APP_SUBTITLE}</div>
            <p>
                This web application predicts whether a protein sequence is
                <b>allergen-like</b> using an LLM-based protein representation
                framework coupled with a downstream machine learning classifier.
            </p>
            <div class="home-card-grid">
                <div class="home-mini-card">
                    <b>Prediction</b><br>
                    Paste or upload FASTA sequences and obtain allergen-like probability scores.
                </div>
                <div class="home-mini-card">
                    <b>Predicted Allergens</b><br>
                    Browse pre-computed candidate allergen-like proteins from fish, crustaceans and molluscs.
                </div>
                <div class="home-mini-card">
                    <b>Explainable AI</b><br>
                    Explore embedding-feature contributions and sequence-region occlusion evidence.
                </div>
            </div>
            <div class="home-note">
                <b>Recommended use:</b> screening and prioritization. Predictions should be interpreted
                as allergen-like / non-allergen-like candidates, not as clinical confirmation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Prediction":
    st.subheader("Input sequences")
    input_mode = st.radio("Choose input mode", ["Paste FASTA", "Upload FASTA file"], horizontal=True)

    example_fasta = ">Seq1\nMSTVHEILCKLSLEGDHSTQAEVSKLAAAFDKYKGKATDKEGKLATVQKAA\n>Seq2\nMADSQKSVKAFNDALEKLRQAVDANNNQ"

    records = []
    if input_mode == "Paste FASTA":
        fasta_text = st.text_area("Paste FASTA here", value=example_fasta, height=220)
        if fasta_text.strip():
            records = parse_fasta_text(fasta_text)
    else:
        uploaded = st.file_uploader("Upload a FASTA file", type=["fasta", "fa", "faa", "txt"])
        if uploaded is not None:
            records = parse_uploaded_fasta(uploaded)

    if records:
        st.success(f"Parsed {len(records)} sequence(s).")
        with st.expander("Preview parsed sequences"):
            preview_df = pd.DataFrame({
                "Protein_ID": [r[0] for r in records],
                "Length": [len(r[1]) for r in records]
            })
            st.dataframe(preview_df, use_container_width=True)

    # Keep prediction output in session_state. Otherwise Streamlit reruns the page
    # when the dropdown changes and the explanation remains tied to the first result.
    if "prediction_pred_df" not in st.session_state:
        st.session_state.prediction_pred_df = None
    if "prediction_X" not in st.session_state:
        st.session_state.prediction_X = None

    run_clicked = st.button("Run prediction", type="primary", disabled=(len(records) == 0))

    if run_clicked:
        try:
            tokenizer, esm_model, clf, device = load_models(model_dir, classifier_path)
            with st.spinner("Embedding sequences and generating predictions..."):
                pred_df, X = predict_sequences(records, tokenizer, esm_model, clf, device)

            pred_df["Output"] = pred_df["Prediction_Probability"].apply(color_class)

            st.session_state.prediction_pred_df = pred_df
            st.session_state.prediction_X = X
            st.session_state.prediction_model_dir = model_dir
            st.session_state.prediction_classifier_path = classifier_path
            st.session_state.selected_explain_protein = pred_df["Protein_ID"].iloc[0] if len(pred_df) else None

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

    pred_df = st.session_state.get("prediction_pred_df", None)
    X = st.session_state.get("prediction_X", None)

    if pred_df is not None and X is not None and not pred_df.empty:
        try:
            tokenizer, esm_model, clf, device = load_models(
                st.session_state.get("prediction_model_dir", model_dir),
                st.session_state.get("prediction_classifier_path", classifier_path),
            )

            display_df = pred_df[["Protein_ID", "Length", "Predicted_Class", "Prediction_Probability", "Output"]].copy()
            display_df["Prediction_Probability"] = display_df["Prediction_Probability"].round(4)

            st.subheader("Prediction results")
            st.dataframe(display_df, use_container_width=True)

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            st.subheader("Explainable AI")
            protein_options = pred_df["Protein_ID"].tolist()
            default_selected = st.session_state.get("selected_explain_protein", protein_options[0])
            default_index = protein_options.index(default_selected) if default_selected in protein_options else 0

            selected_id = st.selectbox(
                "Select a protein for explanation",
                protein_options,
                index=default_index,
                key="selected_explain_protein",
            )

            row_idx = pred_df.index[pred_df["Protein_ID"] == selected_id].tolist()[0]
            selected_seq = pred_df.loc[row_idx, "Protein_Sequence"]
            selected_prob = float(pred_df.loc[row_idx, "Prediction_Probability"])

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Predicted probability:** `{selected_prob:.4f}`")
                st.markdown(f"**Class:** `{pred_df.loc[row_idx, 'Predicted_Class']}`")
                st.markdown("**Global embedding-feature explanation**")
                st.code(get_feature_direction(clf, X[row_idx], top_n=10), language="text")

            with col2:
                st.markdown("**Residue-segment occlusion explanation**")
                st.caption(
                    "Each row shows how much the prediction probability changes when that sequence segment is removed."
                )
                with st.spinner("Generating explanation for selected protein..."):
                    exp_df = segment_occlusion_explanation(selected_seq, tokenizer, esm_model, clf, device, n_segments=8)
                st.dataframe(exp_df, use_container_width=True)
                st.bar_chart(exp_df.set_index("Segment")[["Probability_Drop"]])

            if show_debug:
                st.subheader("Debug")
                st.write(pred_df)
                st.write("Selected row index:", row_idx)

        except Exception as e:
            st.error(f"Explanation failed: {e}")
            st.exception(e)

elif page == "Predicted Allergens":
    st.subheader("Browse pre-computed predicted aquatic allergens")
    st.markdown(
        "Select an aquatic group, choose a species or **All**, and browse candidate allergen-like proteins. "
        "Only a small page of records is displayed at a time for faster viewing."
    )

    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "pred_page" not in st.session_state:
        st.session_state.pred_page = 1

    st.markdown("### Select organism group")
    
    cols = st.columns(3)
    for col, label in zip(cols, CATEGORY_CONFIG.keys()):
        cfg_tmp = CATEGORY_CONFIG[label]
        with col:
            selected_tmp = st.session_state.selected_category == label
            if category_card(label, cfg_tmp, selected_tmp, image_dir):
                st.session_state.selected_category = label
                st.session_state.pred_page = 1
                st.rerun()

    if st.session_state.selected_category is None:
        st.info("Please select Fish, Crustaceans, or Molluscs to view pre-computed predicted allergen-like proteins.")
        st.stop()

    selected_category = st.session_state.selected_category
    cfg = CATEGORY_CONFIG[selected_category]
    category_folder = cfg["folder"]

    st.markdown(f"### {selected_category} predictions")

    files = find_prediction_files(predicted_root, category_folder)

    if not files:
        st.warning(
            f"No prediction CSV files found for {selected_category}. Expected folder: "
            f"{Path(predicted_root) / category_folder}"
        )
        st.markdown(
            "Recommended folder structure:\n\n"
            "```text\n"
            "predicted_allergens/\n"
            "├── fish/\n"
            "├── crustaceans/\n"
            "└── molluscs/\n"
            "```"
        )
    else:
        species_options = ["All"] + sorted(files.keys())
        selected_species = st.selectbox("Select species", species_options, key=f"species_{selected_category}")

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            class_filter = st.selectbox(
                "Class filter",
                ["Predicted allergens only", "All classes", "Predicted non-allergens only"],
                key=f"class_filter_{selected_category}",
            )
        with col_b:
            min_probability = st.slider(
                "Minimum probability",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"min_prob_{selected_category}",
            )
        with col_c:
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0, key=f"rows_{selected_category}")

        with st.spinner("Loading prediction table..."):
            if selected_species == "All":
                df = load_multiple_prediction_csvs(files)
                download_name = f"{selected_category.lower()}_all_species_predictions.csv"
            else:
                df = load_prediction_csv(files[selected_species], selected_species)
                download_name = f"{selected_species.replace(' ', '_')}_predictions.csv"

        if df.empty:
            st.warning("No records available after loading the selected file(s).")
        else:
            filtered_df = filter_prediction_table(df, class_filter, min_probability)

            total_rows = len(filtered_df)
            total_pages = max(1, int(np.ceil(total_rows / rows_per_page)))
            if st.session_state.pred_page > total_pages:
                st.session_state.pred_page = total_pages
            if st.session_state.pred_page < 1:
                st.session_state.pred_page = 1

            n_species = filtered_df["Species"].nunique() if "Species" in filtered_df.columns else 1
            n_allergen = 0
            if "Predicted_Class" in filtered_df.columns:
                n_allergen = (
                    filtered_df["Predicted_Class"].astype(str).str.contains("Allergen", case=False, na=False)
                    & ~filtered_df["Predicted_Class"].astype(str).str.contains("Non", case=False, na=False)
                ).sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Species shown", n_species)
            m2.metric("Filtered records", total_rows)
            m3.metric("Predicted allergens", int(n_allergen))

            start_idx = (st.session_state.pred_page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            page_df = filtered_df.iloc[start_idx:end_idx].copy()

            st.caption(f"Showing records {start_idx + 1 if total_rows else 0}–{min(end_idx, total_rows)} of {total_rows}")
            st.dataframe(page_df, use_container_width=True, hide_index=True)

            # Pagination controls intentionally placed below the table.
            pagination_controls(total_pages, rows_per_page)

            d1, d2 = st.columns([1, 1])
            with d1:
                st.download_button(
                    "Download current page as CSV",
                    data=page_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"page_{st.session_state.pred_page}_{download_name}",
                    mime="text/csv",
                )
            with d2:
                st.download_button(
                    "Download all filtered records as CSV",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name=download_name,
                    mime="text/csv",
                )

            if "Prediction_Probability" in filtered_df.columns and not filtered_df.empty:
                with st.expander("Show probability distribution"):
                    chart_series = filtered_df["Prediction_Probability"].reset_index(drop=True)
                    if len(chart_series) > 5000:
                        chart_series = chart_series.sample(5000, random_state=42).reset_index(drop=True)
                        st.caption("Showing a random sample of 5,000 probabilities for faster plotting.")
                    st.bar_chart(chart_series)

            if show_debug:
                st.subheader("Debug")
                st.write("Files detected:", files)
                st.write("Raw loaded columns:", list(df.columns))

elif page == "Tutorial":
    st.markdown(
        """
        ### Tutorial

        #### 1. Prediction tab
        Provide protein sequences in **FASTA format**.

        Example:
        ```text
        >Protein_A
        MSTVHEILCKLSLEGDHSTQAEVSKLAAAFDKYKGKATDKEGKLATVQKAA
        >Protein_B
        MADSQKSVKAFNDALEKLRQAVDANNNQ
        ```

        Click **Run prediction** to obtain allergen-like probability and class label.

        #### 2. Predicted Allergens tab
        This tab shows pre-computed predictions for aquatic species.

        Recommended folder structure:
        ```text
        predicted_allergens/
        ├── fish/
        │   ├── Salmo salar_predictions.csv
        │   └── Oncorhynchus mykiss_predictions.csv
        ├── crustaceans/
        │   ├── Cancer pagurus_predictions.csv
        │   └── Litopenaeus vannamei_predictions.csv
        └── molluscs/
            ├── Crassostrea gigas_predictions.csv
            └── Haliotis discus_predictions.csv
        ```

        Each CSV should ideally contain:
        - `Protein_ID`
        - `Protein_Sequence`
        - `Protein_Length` or `Length`
        - `Prediction_Probability` or `Prediction_Score`
        - `Predicted_Class`

        #### 3. Interpreting the output
        - `Prediction_Probability >= 0.90`: high-confidence allergen-like candidate
        - `0.70–0.89`: moderate allergen-like candidate
        - `0.50–0.69`: borderline candidate
        - `<0.50`: non-allergen-like

        #### 4. Important note
        Predictions are intended for **screening and prioritization**, not clinical confirmation.
        Candidate allergens should be further evaluated using sequence similarity, epitope mapping,
        structural analysis, proteomics evidence, or experimental assays.
        """
    )

elif page == "Contact":
    st.markdown("## Contact")

    with st.container(border=True):
        st.markdown("**Dr. Sneha Murmu**")
        st.markdown("Scientist (Bioinformatics)")
        st.markdown("Division of Agricultural Bioinformatics")
        st.markdown("ICAR-Indian Agricultural Statistics Research Institute")
        st.markdown("New Delhi – 110012")
        st.markdown("**Email:** [murmu.sneha07@gmail.com](mailto:murmu.sneha07@gmail.com)")