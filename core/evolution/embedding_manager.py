"""
Utility for computing normalized code embeddings using BGE-m3.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class CodeEmbeddingManager:
    """Lazy-loading embedding manager for code snippets."""

    _model_name = "BAAI/bge-m3"
    _model: SentenceTransformer | None = None

    def __init__(self) -> None:
        # text -> embedding
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def _ensure_model(cls) -> None:
        if cls._model is None:
            cls._model = SentenceTransformer(cls._model_name)
            cls._model.eval()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Return L2-normalized embeddings for the provided texts.
        Suitable for FAISS IndexFlatIP / DPP kernels.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        cached_embeddings: list[np.ndarray | None] = []
        missing_texts: list[str] = []

        for text in texts:
            if text in self._cache:
                cached_embeddings.append(self._cache[text])
            else:
                cached_embeddings.append(None)
                
                missing_texts.append(f"code:\n{text}")

        if missing_texts:
            self._ensure_model()

            embeddings = self._model.encode(
                missing_texts,
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,  
            )

            embeddings = embeddings.astype(np.float32)

            idx = 0
            for i, original_text in enumerate(texts):
                if cached_embeddings[i] is None:
                    vec = embeddings[idx]
                    self._cache[original_text] = vec
                    cached_embeddings[i] = vec
                    idx += 1

        result = np.vstack(cached_embeddings)
        return result






# """Utility for computing normalized code embeddings using Jina embeddings."""

# from __future__ import annotations

# from typing import List, Dict

# import numpy as np
# import torch
# from transformers import AutoModel, AutoTokenizer


# class CodeEmbeddingManager:
#     """Lazy-loading embedding manager for code snippets."""

#     _model_name = "jinaai/jina-embeddings-v2-base-code"
#     _tokenizer = None
#     _model = None

#     def __init__(self) -> None:
#         self._cache: Dict[str, np.ndarray] = {}

#     @classmethod
#     def _ensure_model(cls) -> None:
#         if cls._tokenizer is None or cls._model is None:
#             cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
#             cls._model = AutoModel.from_pretrained(cls._model_name)
#             cls._model.eval()

#     def embed_texts(self, texts: List[str]) -> np.ndarray:
#         """Return L2-normalized embeddings for the provided texts."""
#         if not texts:
#             return np.zeros((0, 0), dtype=np.float32)

#         cached_embeddings = []
#         missing_texts = []
#         for text in texts:
#             if text in self._cache:
#                 cached_embeddings.append(self._cache[text])
#             else:
#                 cached_embeddings.append(None)
#                 missing_texts.append(text)

#         if missing_texts:
#             self._ensure_model()
#             inputs = self._tokenizer(
#                 missing_texts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=4096,
#             )
#             with torch.no_grad():
#                 model_output = self._model(**inputs)
#                 embeddings = model_output.last_hidden_state[:, 0, :]
#             embeddings = embeddings.cpu().numpy().astype(np.float32)
#             embeddings = _l2_normalize(embeddings)
#             idx = 0
#             for i, text in enumerate(texts):
#                 if cached_embeddings[i] is None:
#                     vec = embeddings[idx]
#                     self._cache[text] = vec
#                     cached_embeddings[i] = vec
#                     idx += 1

#         result = np.vstack(cached_embeddings)
#         return _l2_normalize(result)


# def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
#     if vectors.size == 0:
#         return vectors.astype(np.float32)
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     norms = np.maximum(norms, 1e-12)
#     normalized = vectors / norms
#     return normalized.astype(np.float32)
