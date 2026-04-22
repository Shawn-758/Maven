"""
rag/bm25_index.py
-----------------
HYBRID COMPONENT A: BM25 Keyword Retrieval

What is BM25?
  BM25 (Best Match 25) is the industry-standard keyword retrieval algorithm.
  It is what powers Elasticsearch, Solr, and Lucene under the hood.
  "25" refers to the 25th iteration of the BM family of ranking functions.

  It's an improved version of TF-IDF (Term Frequency–Inverse Document Frequency).

How BM25 scores a document for a query:
  For each query term t in document d:

      score(t, d) = IDF(t) × [ tf(t,d) × (k1 + 1) ]
                               ─────────────────────────────
                               [ tf(t,d) + k1 × (1 − b + b × |d|/avgdl) ]

  Where:
    tf(t,d)  = how many times term t appears in document d
    IDF(t)   = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
               N = total docs, df(t) = docs containing t
    |d|      = length of document d (in tokens)
    avgdl    = average document length across the corpus
    k1       = term frequency saturation (default 1.5)
               Higher → more credit for repeated terms
    b        = length normalization (default 0.75)
               1.0 = full length norm, 0.0 = no normalization

  Intuition:
    - IDF rewards rare terms (appearing in few docs) over common ones ("the", "is")
    - TF is "saturated": seeing a term 10× gives diminishing returns vs 1×
    - Length normalization: a term in a short focused chunk scores higher
      than the same term buried in a long document

Why BM25 + vector search complement each other:
  BM25 is great at:   exact terms, proper nouns, product names, version numbers
  Vector is great at: paraphrases, synonyms, conceptual queries
  Their failure modes don't overlap — so combining them is almost always better
  than either alone.

Implementation note:
  We use the `rank_bm25` library (pure Python, no heavy deps).
  For production you'd use Elasticsearch or OpenSearch instead,
  which handle scaling, persistence, and incremental updates.
"""

import math
import re
import pickle
from typing import Optional
from ..chunker import Chunk


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.

    For production, use a proper stemmer/lemmatizer (e.g. nltk's PorterStemmer)
    so "running" and "run" match the same token. Here we keep it simple.

    Trade-off:
      Simple tokenizer → fast, no dependencies, misses morphological variants
      NLTK stemmer     → better recall, adds ~50ms/query, requires download
    """
    text = text.lower()
    # Split on non-alphanumeric characters, filter empty strings
    tokens = re.split(r'[^a-z0-9]+', text)
    return [t for t in tokens if t and len(t) > 1]  # drop single chars


class BM25Index:
    """
    An in-memory BM25 index over a list of Chunks.

    The index is built once at ingestion time and queried at retrieval time.
    It stores:
      - A corpus of tokenized chunks
      - IDF scores for every unique token
      - Per-document term frequencies
      - References back to the original Chunk objects

    Persistence:
      Call save(path) / load(path) to serialize to disk with pickle.
      In production you'd use a proper search engine instead.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Parameters
        ----------
        k1 : float
            Term frequency saturation factor. 1.2–2.0 is typical.
            Lower k1 = diminishing returns kick in faster (good for short queries).
        b : float
            Length normalization. 0 = no normalization, 1 = full normalization.
            0.75 is the BM25 paper's original recommendation.
        """
        self.k1 = k1
        self.b = b

        # Core state — populated by build()
        self.chunks: list[Chunk] = []
        self.corpus_tokens: list[list[str]] = []   # tokenized text per chunk
        self.doc_freqs: dict[str, int] = {}        # how many docs each term appears in
        self.idf: dict[str, float] = {}            # IDF score per term
        self.doc_len: list[int] = []               # token count per chunk
        self.avgdl: float = 0.0                    # average document length
        self._built = False

    def build(self, chunks: list[Chunk]) -> None:
        """
        Index a list of chunks. O(N × L) where N=chunks, L=avg chunk length.

        Call this once after ingestion. For incremental updates (new docs added
        later), you'd need to rebuild or use an external search engine.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)

        # Count document frequency for every unique term
        self.doc_freqs = {}
        for tokens in self.corpus_tokens:
            seen_in_doc = set(tokens)  # count each term once per document
            for token in seen_in_doc:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Precompute IDF for every term
        N = len(chunks)
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # The +0.5 smoothing prevents IDF=0 for terms in every document
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        self._built = True
        vocab_size = len(self.doc_freqs)
        print(
            f"  BM25 index built: {len(chunks)} chunks, "
            f"{vocab_size:,} unique terms, avgdl={self.avgdl:.1f} tokens"
        )

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """
        Retrieve the top-k chunks most relevant to the query.

        Returns a list of (Chunk, bm25_score) tuples sorted by score descending.
        Chunks with score=0 (no keyword overlap) are excluded.

        Parameters
        ----------
        query : str
            The raw user query (will be tokenized internally).
        k : int
            Number of results to return.
        """
        if not self._built:
            raise RuntimeError("Call build() before search()")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._score_all(query_tokens)

        # Sort by score, filter zeros, return top k
        scored_chunks = [
            (self.chunks[i], scores[i])
            for i in range(len(self.chunks))
            if scores[i] > 0
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:k]

    def _score_all(self, query_tokens: list[str]) -> list[float]:
        """
        Compute BM25 score for every chunk in the corpus.
        This is O(Q × N) where Q=query terms, N=chunks.

        For large corpora (>100k chunks) you'd use an inverted index
        to only score documents that contain at least one query term.
        """
        scores = [0.0] * len(self.chunks)

        for token in query_tokens:
            if token not in self.idf:
                continue  # unseen term: contributes nothing

            idf_score = self.idf[token]

            for doc_idx, tokens in enumerate(self.corpus_tokens):
                # Term frequency in this document
                tf = tokens.count(token)
                if tf == 0:
                    continue

                # BM25 formula
                dl = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf_score * numerator / denominator

        return scores

    def save(self, path: str) -> None:
        """Serialize the index to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"  BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a previously saved index."""
        index = cls()
        with open(path, "rb") as f:
            index.__dict__.update(pickle.load(f))
        print(f"  BM25 index loaded from {path} ({len(index.chunks)} chunks)")
        return index

    @property
    def size(self) -> int:
        return len(self.chunks)