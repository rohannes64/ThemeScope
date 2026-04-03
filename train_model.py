"""
train_model.py — ThemeScope (NMF Edition)
==========================================
Model      : NMF (Non-negative Matrix Factorization)
Objective  : Minimize KL divergence between X and WH
Vectorizer : TF-IDF

Datasets (downloaded ONCE, cached locally as parquet):
  - arXiv       : gfissore/arxiv-abstracts-2021   → 8 domains
  - PubMed      : qiaojin/PubMedQA                → Medical & Clinical Research
  - ClimateCheck: rabuahmad/climatecheck_publications_corpus → Climate & Environmental

Run:
    pip install -r requirements.txt
    python train_model.py        # first run downloads + trains
    python train_model.py        # subsequent runs use local cache, skip download
"""

import os, pickle, json, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR        = "data_cache"
ARXIV_CACHE      = os.path.join(CACHE_DIR, "arxiv.parquet")
PUBMED_CACHE     = os.path.join(CACHE_DIR, "pubmed.parquet")
CLIMATE_CACHE    = os.path.join(CACHE_DIR, "climate.parquet")
CORPUS_CACHE     = os.path.join(CACHE_DIR, "corpus.parquet")   # final merged corpus

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_PER_DOMAIN  = 1000   # abstracts per domain
N_TOPICS      = 10
N_COMPONENTS  = 10

# ── Domain definitions ───────────────────────────────────────────────────────
DOMAIN_NAMES = [
    "Machine Learning & AI",
    "Bioinformatics & Genomics",
    "Climate & Environmental Science",
    "Quantum Computing & Physics",
    "Medical & Clinical Research",
    "Robotics & Autonomous Systems",
    "Economics & Finance",
    "Materials Science & Nanotechnology",
    "Social Sciences & Psychology",
    "Astronomy & Astrophysics",
]

DOMAIN_COLORS = {
    "Machine Learning & AI":               "#6366f1",
    "Bioinformatics & Genomics":           "#10b981",
    "Climate & Environmental Science":     "#0ea5e9",
    "Quantum Computing & Physics":         "#8b5cf6",
    "Medical & Clinical Research":         "#f43f5e",
    "Robotics & Autonomous Systems":       "#f59e0b",
    "Economics & Finance":                 "#14b8a6",
    "Materials Science & Nanotechnology":  "#ec4899",
    "Social Sciences & Psychology":        "#84cc16",
    "Astronomy & Astrophysics":            "#f97316",
}

# arXiv category → domain (only for arXiv-sourced domains)
ARXIV_DOMAIN_CATEGORIES = {
    "Machine Learning & AI": [
        "cs.LG", "cs.AI", "cs.NE", "stat.ML"
    ],
    "Bioinformatics & Genomics": [
        "q-bio.GN", "q-bio.QM", "q-bio.MN", "q-bio.BM"
    ],
    "Quantum Computing & Physics": [
        "quant-ph", "cond-mat.mes-hall", "cond-mat.str-el"
    ],
    "Robotics & Autonomous Systems": [
        "cs.RO", "cs.SY", "eess.SY"
    ],
    "Economics & Finance": [
        "econ.GN", "econ.EM", "q-fin.GN", "q-fin.TR", "q-fin.EC"
    ],
    "Materials Science & Nanotechnology": [
        "cond-mat.mtrl-sci", "cond-mat.supr-con", "cond-mat.soft"
    ],
    "Social Sciences & Psychology": [
        "cs.CY", "cs.HC", "cs.SI"
    ],
    "Astronomy & Astrophysics": [
        "astro-ph.GA", "astro-ph.SR", "astro-ph.CO", "astro-ph.HE"
    ],
}

# Seed words for topic→domain alignment (Hungarian algorithm)
DOMAIN_SEEDS = {
    "Machine Learning & AI": [
        "neural", "deep learning", "convolutional", "gradient", "classification",
        "supervised", "transformer", "attention", "reinforcement", "representations",
        "accuracy", "training", "autoencoder", "fine-tuning", "generative",
    ],
    "Bioinformatics & Genomics": [
        "gene", "sequencing", "genomic", "expression", "dna", "rna",
        "differential", "microbiome", "omics", "pathway", "protein",
        "crispr", "transcriptome", "methylation", "variant",
    ],
    "Climate & Environmental Science": [
        "climate", "warming", "carbon", "co2", "emissions", "ocean",
        "atmospheric", "sea ice", "permafrost", "aerosol", "temperature",
        "precipitation", "drought", "ecosystem", "greenhouse",
    ],
    "Quantum Computing & Physics": [
        "quantum", "qubit", "entanglement", "decoherence", "photonic",
        "quantum circuit", "quantum error", "superposition", "topological",
        "superconducting", "quantum key", "quantum advantage",
    ],
    "Medical & Clinical Research": [
        "patient", "clinical", "treatment", "disease", "therapeutic",
        "drug", "randomized", "trial", "cancer", "immune", "diagnosis",
        "surgery", "infection", "fracture", "hospital", "cohort",
    ],
    "Robotics & Autonomous Systems": [
        "robot", "robotic", "autonomous", "lidar", "localization",
        "manipulation", "drone", "navigation", "sensor fusion", "planning",
        "quadrotor", "actuator", "grasping", "trajectory",
    ],
    "Economics & Finance": [
        "market", "economic", "labor", "income", "wage", "policy",
        "financial", "trading", "inequality", "monetary", "gdp",
        "inflation", "fiscal", "equity", "behavioral economics",
    ],
    "Materials Science & Nanotechnology": [
        "atomic", "spectroscopy", "microscopy", "synthesis", "nanocomposite",
        "perovskite", "electrolyte", "graphene", "thin film", "crystalline",
        "semiconductor", "alloy", "corrosion", "polymer", "nanomaterial",
    ],
    "Social Sciences & Psychology": [
        "social", "cognitive", "behavior", "psychological", "cultural",
        "wellbeing", "bias", "socioeconomic", "mental health", "intervention",
        "survey", "attitude", "emotion", "personality",
    ],
    "Astronomy & Astrophysics": [
        "stellar", "neutron star", "dark matter", "dark energy", "galaxy",
        "cosmological", "gravitational", "redshift", "black hole", "supernova",
        "quasar", "exoplanet", "pulsar", "nebula",
    ],
}

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\$[^$]*\$", " ", text)            # strip LaTeX math
    text = re.sub(r"\\[a-z]+\{[^}]*\}", " ", text)    # strip LaTeX commands
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download datasets (once) and cache locally
# ══════════════════════════════════════════════════════════════════════════════

def download_arxiv():
    """Download arXiv Parquet shards and cache the filtered abstracts."""
    if os.path.exists(ARXIV_CACHE):
        print("  [arXiv] Cache found — skipping download")
        return pd.read_parquet(ARXIV_CACHE)

    print("  [arXiv] Downloading Parquet shards from HuggingFace...")
    BASE_URL = (
        "https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/{:04d}.parquet"
    )
    TOTAL_SHARDS = 19

    # Build reverse map: category → domain
    cat_to_domain = {}
    for domain, cats in ARXIV_DOMAIN_CATEGORIES.items():
        for cat in cats:
            cat_to_domain[cat] = domain

    buckets = {d: [] for d in ARXIV_DOMAIN_CATEGORIES}

    for shard_idx in range(TOTAL_SHARDS):
        if all(len(b) >= N_PER_DOMAIN for b in buckets.values()):
            break

        url = BASE_URL.format(shard_idx)
        print(f"    Shard {shard_idx+1}/{TOTAL_SHARDS}...", end=" ", flush=True)
        try:
            df = pd.read_parquet(url, columns=["abstract", "categories"])
        except Exception as e:
            print(f"SKIP ({e})")
            continue
        print(f"{len(df):,} rows")

        for _, row in df.iterrows():
            abstract = clean(row.get("abstract", "") or "")
            if len(abstract.split()) < 40:
                continue

            cats = row.get("categories") or []
            if isinstance(cats, str):
                cats = cats.split()

            for cat in cats:
                domain = cat_to_domain.get(cat)
                if domain and len(buckets[domain]) < N_PER_DOMAIN:
                    buckets[domain].append({"abstract": abstract, "domain": domain})
                    break

    rows = [r for bucket in buckets.values() for r in bucket]
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(ARXIV_CACHE, index=False)
    print(f"  [arXiv] Saved {len(df_out):,} abstracts → {ARXIV_CACHE}")
    for d, b in buckets.items():
        print(f"    {d:45s}: {len(b):,}")
    return df_out


def download_pubmed():
    """Download PubMedQA (pure Parquet) for Medical & Clinical Research."""
    if os.path.exists(PUBMED_CACHE):
        print("  [PubMed] Cache found — skipping download")
        return pd.read_parquet(PUBMED_CACHE)

    print("  [PubMed] Downloading qiaojin/PubMedQA (pqa_artificial)...")
    # pqa_artificial has 211k entries — plenty
    URL = (
        "https://huggingface.co/datasets/qiaojin/PubMedQA"
        "/resolve/refs%2Fconvert%2Fparquet/pqa_artificial/train/0000.parquet"
    )
    try:
        df = pd.read_parquet(URL)
    except Exception as e:
        print(f"  [PubMed] ERROR: {e}")
        return pd.DataFrame(columns=["abstract", "domain"])

    print(f"  [PubMed] Downloaded {len(df):,} rows. Columns: {list(df.columns)}")

    # PubMedQA has a 'context' column containing the abstract paragraphs
    rows = []
    for _, row in df.iterrows():
        # context is a dict with 'contexts' key (list of sentences)
        ctx = row.get("context", {})
        if isinstance(ctx, dict):
            sentences = ctx.get("contexts", [])
            text = " ".join(sentences) if sentences else ""
        else:
            text = str(ctx)

        text = clean(text)
        if len(text.split()) >= 40:
            rows.append({"abstract": text, "domain": "Medical & Clinical Research"})
        if len(rows) >= N_PER_DOMAIN:
            break

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(PUBMED_CACHE, index=False)
    print(f"  [PubMed] Saved {len(df_out):,} abstracts → {PUBMED_CACHE}")
    return df_out


def download_climate():
    """Download ClimateCheck corpus for Climate & Environmental Science."""
    if os.path.exists(CLIMATE_CACHE):
        print("  [Climate] Cache found — skipping download")
        return pd.read_parquet(CLIMATE_CACHE)

    print("  [Climate] Downloading rabuahmad/climatecheck_publications_corpus...")
    # This dataset has 394k climate abstracts — direct Parquet
    BASE_URL = (
        "https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/{:04d}.parquet"
    )

    rows = []
    for shard_idx in range(10):   # try up to 10 shards
        if len(rows) >= N_PER_DOMAIN:
            break
        url = BASE_URL.format(shard_idx)
        print(f"    Shard {shard_idx+1}...", end=" ", flush=True)
        try:
            df = pd.read_parquet(url)
            print(f"{len(df):,} rows | cols: {list(df.columns)[:5]}")
        except Exception as e:
            print(f"STOP ({e})")
            break

        # Find the abstract column
        abstract_col = None
        for col in ["abstract", "text", "body", "content", "Abstract"]:
            if col in df.columns:
                abstract_col = col
                break
        if abstract_col is None:
            print(f"    No abstract column found in {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            text = clean(str(row.get(abstract_col, "") or ""))
            if len(text.split()) >= 40:
                rows.append({"abstract": text, "domain": "Climate & Environmental Science"})
            if len(rows) >= N_PER_DOMAIN:
                break

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(CLIMATE_CACHE, index=False)
    print(f"  [Climate] Saved {len(df_out):,} abstracts → {CLIMATE_CACHE}")
    return df_out


def build_corpus():
    """Merge all sources into a single corpus cache."""
    if os.path.exists(CORPUS_CACHE):
        print("📂 Corpus cache found — loading from disk (delete data_cache/ to re-download)\n")
        df = pd.read_parquet(CORPUS_CACHE)
        print(f"   {len(df):,} abstracts loaded")
        for d in DOMAIN_NAMES:
            n = len(df[df.domain == d])
            print(f"   {d:45s}: {n:,}")
        print()
        return df["abstract"].tolist(), df["domain"].tolist()

    print("⬇  Downloading datasets (this happens ONCE — cached after)...\n")
    df_arxiv   = download_arxiv()
    df_pubmed  = download_pubmed()
    df_climate = download_climate()

    df_all = pd.concat([df_arxiv, df_pubmed, df_climate], ignore_index=True)
    df_all = df_all.dropna(subset=["abstract"])
    df_all = df_all[df_all["abstract"].str.split().str.len() >= 40]

    df_all.to_parquet(CORPUS_CACHE, index=False)
    print(f"\n✅ Corpus saved → {CORPUS_CACHE} ({len(df_all):,} abstracts total)")
    print("   Next run will load from cache instantly.\n")

    return df_all["abstract"].tolist(), df_all["domain"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train NMF with KL divergence
# ══════════════════════════════════════════════════════════════════════════════

EXTRA_STOPS = [
    "using", "based", "used", "use", "show", "shows", "shown", "paper",
    "study", "studies", "research", "approach", "method", "methods",
    "result", "results", "propose", "proposed", "present", "presented",
    "data", "analysis", "model", "models", "new", "different", "large",
    "high", "higher", "low", "also", "including", "number", "provide",
    "demonstrate", "improve", "improved", "compare", "achieve", "achieved",
    "across", "between", "within", "apply", "applied", "dataset", "test",
    "performance", "framework", "algorithm", "algorithms", "task", "tasks",
    "system", "systems", "information", "work", "enable", "given", "set",
    "well", "arxiv", "preprint", "abstract", "figure", "table",
]


def train_nmf(abstracts):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stops = list(ENGLISH_STOP_WORDS) + EXTRA_STOPS

    print("🔤 Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_df=0.90,
        min_df=5,
        max_features=8000,
        stop_words=stops,
        ngram_range=(1, 2),
        sublinear_tf=True,      # log(1+tf) — better for NMF
    )
    X = vectorizer.fit_transform(abstracts)
    print(f"   Matrix shape: {X.shape}  (docs × vocab)")
    print(f"   Vocabulary size: {X.shape[1]:,} terms\n")

    print("⚙  Training NMF (KL divergence, multiplicative updates)...")
    print("   This runs until convergence — typically 1-3 minutes.\n")
    nmf = NMF(
        n_components=N_TOPICS,
        beta_loss="kullback-leibler",   # KL divergence objective
        solver="mu",                     # multiplicative updates (required for KL)
        max_iter=400,
        random_state=42,
        alpha_W=0.1,                     # L1 regularization on W
        alpha_H=0.1,                     # L1 regularization on H
        l1_ratio=0.5,
        verbose=1,
    )
    W = nmf.fit_transform(X)
    print(f"\n   Converged after {nmf.n_iter_} iterations")
    print(f"   Reconstruction error (KL): {nmf.reconstruction_err_:.4f}\n")

    return vectorizer, nmf, W


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Align topics to domain labels (Hungarian algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def align_topics(nmf, vectorizer):
    feature_names = list(vectorizer.get_feature_names_out())
    feat_idx = {w: i for i, w in enumerate(feature_names)}

    score_matrix = np.zeros((N_TOPICS, len(DOMAIN_NAMES)))
    for t_idx, topic_vec in enumerate(nmf.components_):
        topic_norm = topic_vec / (topic_vec.sum() + 1e-10)
        for d_idx, domain in enumerate(DOMAIN_NAMES):
            score_matrix[t_idx, d_idx] = sum(
                topic_norm[feat_idx[s]]
                for s in DOMAIN_SEEDS[domain]
                if s in feat_idx
            )

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    topic_to_domain = {str(int(t)): DOMAIN_NAMES[d] for t, d in zip(row_ind, col_ind)}

    print("🗺  Topic → Domain alignment:")
    for t in range(N_TOPICS):
        d   = topic_to_domain[str(t)]
        scr = score_matrix[t, DOMAIN_NAMES.index(d)] * 1000
        print(f"   Topic {t:2d} → {d:45s} (score={scr:.1f})")

    return topic_to_domain


def get_top_keywords(nmf, vectorizer, topic_idx, n=10):
    feat = vectorizer.get_feature_names_out()
    top  = nmf.components_[topic_idx].argsort()[:-n-1:-1]
    return [feat[i] for i in top]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate coherence
# ══════════════════════════════════════════════════════════════════════════════

def coherence_score(nmf, vectorizer, abstracts, topic_to_domain, n_top=10):
    """
    Compute pointwise mutual information (PMI) based coherence score.
    Higher = more coherent topics.
    """
    from sklearn.feature_extraction.text import CountVectorizer as CV
    cv = CV(vocabulary=vectorizer.vocabulary_, binary=True)
    doc_term = cv.fit_transform(abstracts)
    n_docs = doc_term.shape[0]

    feat = vectorizer.get_feature_names_out()
    scores = []

    for t_idx in range(N_TOPICS):
        top_idx  = nmf.components_[t_idx].argsort()[:-n_top-1:-1]
        top_words = top_idx.tolist()
        score = 0
        pairs = 0
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                wi, wj = top_words[i], top_words[j]
                co  = (doc_term[:, wi].toarray().flatten() *
                       doc_term[:, wj].toarray().flatten()).sum()
                pi  = doc_term[:, wi].sum()
                pj  = doc_term[:, wj].sum()
                if co > 0 and pi > 0 and pj > 0:
                    score += np.log((co * n_docs) / (pi * pj + 1e-10))
                pairs += 1
        scores.append(score / pairs if pairs > 0 else 0)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Accuracy check & save
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(W, labels, topic_to_domain):
    correct = 0
    for i, true_label in enumerate(labels):
        top_topic  = int(np.argmax(W[i]))
        pred_label = topic_to_domain[str(top_topic)]
        if pred_label == true_label:
            correct += 1
    acc = correct / len(labels) * 100
    print(f"\n📊 Training set accuracy: {correct}/{len(labels)} = {acc:.1f}%")
    return acc


def save_artifacts(vectorizer, nmf, topic_to_domain, topic_keywords, coherence_scores):
    with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    with open("nmf_model.pkl",  "wb") as f: pickle.dump(nmf, f)
    with open("topic_mapping.json", "w") as f:
        json.dump({
            "topic_to_domain":  topic_to_domain,
            "topic_keywords":   topic_keywords,
            "domain_colors":    DOMAIN_COLORS,
            "coherence_scores": {
                topic_to_domain[str(t)]: round(coherence_scores[t], 4)
                for t in range(N_TOPICS)
            },
        }, f, indent=2)
    print("\n💾 Saved: vectorizer.pkl, nmf_model.pkl, topic_mapping.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("  ThemeScope — NMF Topic Model (KL Divergence)")
    print("=" * 60 + "\n")

    # 1. Load corpus (download once, cache forever)
    abstracts, labels = build_corpus()

    # 2. Train NMF
    vectorizer, nmf, W = train_nmf(abstracts)

    # 3. Align topics to domain labels
    print()
    topic_to_domain = align_topics(nmf, vectorizer)

    # 4. Get keywords per topic
    topic_keywords = {
        str(t): get_top_keywords(nmf, vectorizer, t, n=10)
        for t in range(N_TOPICS)
    }

    print("\n📝 Top keywords per topic:")
    for t in range(N_TOPICS):
        domain = topic_to_domain[str(t)]
        kws    = ", ".join(topic_keywords[str(t)][:6])
        print(f"   {domain:45s} | {kws}")

    # 5. Coherence scores
    print("\n📐 Computing topic coherence scores...")
    coherence_scores = coherence_score(nmf, vectorizer, abstracts, topic_to_domain)
    avg_coherence = np.mean(coherence_scores)
    print(f"   Average coherence: {avg_coherence:.4f}")
    for t in range(N_TOPICS):
        print(f"   Topic {t:2d} [{topic_to_domain[str(t)][:30]:30s}]: {coherence_scores[t]:.4f}")

    # 6. Accuracy
    evaluate(W, labels, topic_to_domain)

    # 7. Save
    save_artifacts(vectorizer, nmf, topic_to_domain, topic_keywords, coherence_scores)

    print("\n🚀 Done! Run `python app.py` to start the web app.")


if __name__ == "__main__":
    train()
