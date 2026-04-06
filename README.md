# ThemeScope — Research Theme Discovery (NMF Edition) 

Topic modeling web app using **NMF with KL Divergence** on real research abstracts.

## Model

| Property | Detail |
|----------|--------|
| Model | Non-negative Matrix Factorization (NMF) |
| Objective | Minimize KL divergence: `KL(X \|\| WH)` |
| Solver | Multiplicative Updates (`mu`) |
| Vectorizer | TF-IDF (sublinear, bigrams, 8000 features) |
| Topics | 10 domains |
| Training docs | ~10,000 abstracts |

## Datasets

| Source | Domain | Size |
|--------|--------|------|
| `gfissore/arxiv-abstracts-2021` | 8 domains (ML, Bio, Quantum, Robotics, Econ, Materials, Social, Astro) | 2M papers |
| `qiaojin/PubMedQA` | Medical & Clinical Research | 211k papers |
| `rabuahmad/climatecheck_publications_corpus` | Climate & Environmental Science | 394k papers |

## Setup

```bash
pip install -r requirements.txt

# First run: downloads datasets, trains model (~5-10 min)
python train_model.py

# Every run after: loads from cache instantly
python train_model.py   # uses data_cache/ — no download

# Start the web app
python app.py
# Open http://localhost:5000
```

## File Structure

```
topic_app/
├── app.py                  ← Flask backend
├── train_model.py          ← NMF training pipeline
├── requirements.txt
├── vectorizer.pkl          ← saved TF-IDF vectorizer
├── nmf_model.pkl           ← saved NMF model
├── topic_mapping.json      ← topic→domain labels + keywords + coherence
├── data_cache/             ← downloaded datasets (auto-created)
│   ├── arxiv.parquet       ← arXiv abstracts cache
│   ├── pubmed.parquet      ← PubMed abstracts cache
│   ├── climate.parquet     ← Climate abstracts cache
│   └── corpus.parquet      ← merged corpus cache
└── templates/
    └── index.html
```

## Re-downloading data

Delete the cache folder and retrain:
```bash
rmdir /s data_cache    # Windows
rm -rf data_cache      # Mac/Linux
python train_model.py
```
