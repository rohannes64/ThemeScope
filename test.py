import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
    "well", "time", "state", "order", "field", "structure", "non", "problem",
    "energy", "first", "two", "one", "three", "general", "form", "case",
    "consider", "type", "function", "value", "point", "class", "space",
    "theory", "property", "process", "number",
]

stops = list(ENGLISH_STOP_WORDS) + EXTRA_STOPS

vectorizer = CountVectorizer(
    max_df=0.85,
    min_df=15,
    max_features=5000,
    stop_words=stops,
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
)

df = pd.read_parquet("data_cache/corpus.parquet")
abstracts = df["abstract"].tolist()

vectorizer = CountVectorizer(max_df=0.85, min_df=15, max_features=5000,
                              ngram_range=(1,2), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vectorizer.fit_transform(abstracts)

nmf = NMF(n_components=10, beta_loss="kullback-leibler", solver="mu",
          max_iter=400, tol=1e-6, random_state=42,
          alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,  # zero regularization
          init="nndsvda", verbose=1)

W = nmf.fit_transform(X)

feat = vectorizer.get_feature_names_out()
for i, topic in enumerate(nmf.components_):
    top = topic.argsort()[:-8:-1]
    print(f"Topic {i}: {[feat[j] for j in top]}")