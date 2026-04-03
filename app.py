"""
app.py — ThemeScope Flask backend (NMF edition)
"""
import pickle, json, re, os
from flask import Flask, request, jsonify, render_template

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    needed = ["vectorizer.pkl", "nmf_model.pkl", "topic_mapping.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(BASE, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing: {missing}. Run `python train_model.py` first."
        )
    with open(os.path.join(BASE, "vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    with open(os.path.join(BASE, "nmf_model.pkl"), "rb") as f:
        nmf = pickle.load(f)
    with open(os.path.join(BASE, "topic_mapping.json")) as f:
        m = json.load(f)
    return vec, nmf, m

vectorizer, nmf, mapping = load_artifacts()
TOPIC_TO_DOMAIN  = mapping["topic_to_domain"]
TOPIC_KEYWORDS   = mapping["topic_keywords"]
DOMAIN_COLORS    = mapping["domain_colors"]
COHERENCE_SCORES = mapping.get("coherence_scores", {})

DOMAIN_ICONS = {
    "Machine Learning & AI":               "🤖",
    "Bioinformatics & Genomics":           "🧬",
    "Climate & Environmental Science":     "🌍",
    "Quantum Computing & Physics":         "⚛️",
    "Medical & Clinical Research":         "🏥",
    "Robotics & Autonomous Systems":       "🦾",
    "Economics & Finance":                 "📈",
    "Materials Science & Nanotechnology":  "⚗️",
    "Social Sciences & Psychology":        "🧠",
    "Astronomy & Astrophysics":            "🔭",
}

def clean_text(text: str) -> str:
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-z]+\{[^}]*\}", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    body     = request.get_json(force=True)
    abstract = body.get("abstract", "").strip()

    if not abstract:
        return jsonify({"error": "Please provide an abstract."}), 400
    if len(abstract.split()) < 15:
        return jsonify({"error": "Abstract too short — please provide at least 15 words."}), 400

    # TF-IDF → NMF transform
    X = vectorizer.transform([clean_text(abstract)])
    W = nmf.transform(X)[0]          # topic weight vector

    # Normalize to sum to 100%
    total = W.sum() or 1.0
    W_norm = W / total

    themes = []
    for t_idx, weight in enumerate(W_norm):
        if weight < 0.03:
            continue
        domain = TOPIC_TO_DOMAIN.get(str(t_idx), f"Topic {t_idx}")
        themes.append({
            "topic_id": int(t_idx),
            "name":     domain,
            "icon":     DOMAIN_ICONS.get(domain, "🔬"),
            "weight":   round(float(weight) * 100, 1),
            "keywords": TOPIC_KEYWORDS.get(str(t_idx), []),
            "color":    DOMAIN_COLORS.get(domain, "#6366f1"),
            "coherence": COHERENCE_SCORES.get(domain, 0),
        })

    themes.sort(key=lambda x: x["weight"], reverse=True)

    return jsonify({
        "themes":     themes,
        "primary":    themes[0] if themes else None,
        "word_count": len(abstract.split()),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
