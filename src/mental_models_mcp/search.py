"""Embedding-based semantic search for mental models."""

import re
from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# Decision type → discipline boost factors
DECISION_TYPE_BOOSTS: dict[str, dict[str, float]] = {
    "deal_evaluation": {
        "Financial Theory": 1.3,
        "Investing": 1.3,
        "Probability": 1.3,
    },
    "negotiation": {
        "Game Theory": 1.5,
        "Behavioral Economics": 1.5,
    },
    "resource_allocation": {
        "Economics": 1.3,
        "Elementary Models": 1.3,
    },
    "investment": {
        "Investing": 1.4,
        "Financial Theory": 1.4,
        "Probability": 1.4,
    },
    "hiring": {
        "Behavioral Economics": 1.3,
        "Philosophy": 1.3,
    },
    "strategy": {
        "Elementary Models": 1.3,
        "Game Theory": 1.3,
        "Economics": 1.3,
    },
    "general": {},
}


def _simple_tokenize(text: str) -> list[str]:
    """Basic tokenizer for TF-IDF fallback."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class ModelSearch:
    """Semantic search over pre-computed model embeddings.

    Uses ONNX Runtime for query embedding when available,
    falls back to TF-IDF cosine similarity otherwise.
    """

    def __init__(
        self,
        parsed_models: list[dict],
        embeddings_path: Optional[Path] = None,
        onnx_model_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
    ):
        self._models = parsed_models
        self._model_ids = [m["id"] for m in parsed_models]
        self._model_disciplines = [m["discipline"] for m in parsed_models]

        embeddings_path = embeddings_path or DATA_DIR / "embeddings.npz"
        onnx_model_path = onnx_model_path or DATA_DIR / "model.onnx"
        tokenizer_path = tokenizer_path or DATA_DIR / "tokenizer"

        self._embeddings: Optional[np.ndarray] = None
        self._session = None
        self._tokenizer = None
        self._use_tfidf = False

        # Try to load pre-computed embeddings
        if embeddings_path.exists():
            data = np.load(embeddings_path, allow_pickle=True)
            self._embeddings = data["embeddings"].astype(np.float32)

        # Try to load ONNX model for runtime query embedding
        if onnx_model_path.exists():
            try:
                from onnxruntime import InferenceSession
                self._session = InferenceSession(str(onnx_model_path))
                # Load tokenizer
                if tokenizer_path.exists():
                    from tokenizers import Tokenizer
                    self._tokenizer = Tokenizer.from_file(
                        str(tokenizer_path / "tokenizer.json")
                    )
            except (ImportError, Exception):
                self._session = None

        # Fallback: build TF-IDF if no ONNX or no embeddings
        if self._embeddings is None or self._session is None:
            self._use_tfidf = True
            self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        """Build a simple TF-IDF index for fallback search."""
        corpus = []
        for m in self._models:
            text = self._model_text(m)
            corpus.append(text)

        # Build vocabulary
        vocab: dict[str, int] = {}
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            tokens = set(_simple_tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                if token not in vocab:
                    vocab[token] = len(vocab)

        n_docs = len(corpus)
        n_vocab = len(vocab)

        # Build TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for i, doc in enumerate(corpus):
            tokens = _simple_tokenize(doc)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                if t in vocab:
                    idf = np.log(n_docs / (1 + doc_freq.get(t, 0)))
                    tfidf_matrix[i, vocab[t]] = count * idf

        # Normalize rows
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf_matrix /= norms

        self._tfidf_matrix = tfidf_matrix
        self._vocab = vocab
        self._doc_freq = doc_freq
        self._n_docs = n_docs

    def _model_text(self, model: dict) -> str:
        """Composite text for embedding/search."""
        parts = [
            model.get("name", ""),
            model.get("discipline", ""),
            model.get("summary", ""),
            model.get("heart", "") or "",
        ]
        return ". ".join(p for p in parts if p)

    def _embed_query_onnx(self, query: str) -> np.ndarray:
        """Embed a query using the ONNX model."""
        encoding = self._tokenizer.encode(query)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # Mean pooling over token embeddings
        token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.sum(mask_expanded, axis=1)
        embedding = summed / counts
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()

    def _embed_query_tfidf(self, query: str) -> np.ndarray:
        """Embed a query using TF-IDF."""
        tokens = _simple_tokenize(query)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, count in tf.items():
            if t in self._vocab:
                idf = np.log(self._n_docs / (1 + self._doc_freq.get(t, 0)))
                vec[self._vocab[t]] = count * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(
        self,
        query: str,
        top_k: int = 5,
        discipline_filter: Optional[str] = None,
        decision_type: Optional[str] = None,
    ) -> list[dict]:
        """Search for models matching the query.

        Returns a list of dicts with model data and relevance score.
        """
        if self._use_tfidf:
            query_vec = self._embed_query_tfidf(query)
            scores = self._tfidf_matrix @ query_vec
        else:
            query_vec = self._embed_query_onnx(query)
            scores = self._embeddings @ query_vec

        # Apply decision_type boosting
        if decision_type and decision_type in DECISION_TYPE_BOOSTS:
            boosts = DECISION_TYPE_BOOSTS[decision_type]
            for i, disc in enumerate(self._model_disciplines):
                if disc in boosts:
                    scores[i] *= boosts[disc]

        # Apply discipline filter
        if discipline_filter:
            lower_filter = discipline_filter.lower()
            for i, disc in enumerate(self._model_disciplines):
                if disc.lower() != lower_filter:
                    scores[i] = -1.0

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            model = self._models[idx].copy()
            model["relevance_score"] = float(scores[idx])
            results.append(model)

        return results
