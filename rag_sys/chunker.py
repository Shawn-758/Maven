"""
  Split a long Document into smaller Chunk objects that fit within the
  embedding model's context window and are semantically coherent.
"""

from dataclasses import dataclass, field
from typing import Optional
from .ingestion import Document


@dataclass
class Chunk:
    """
   composition:
    text       : the actual chunk text
    metadata   : inherited from parent Document + chunk-specific fields
    chunk_id   : unique id (used as the vector DB record ID)
    doc_id     : links back to the parent Document
    chunk_index: position of this chunk within the document (for re-ordering)
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0

    def __post_init__(self):
        if self.chunk_id is None:
            import uuid
            self.chunk_id = str(uuid.uuid4())


class TextChunker:
    """
    Splits Documents into Chunks

    Parameters
    ----------
    chunk_size : int
        Target size of each chunk in CHARACTERS.

    overlap : int
        How many characters each chunk shares with the next.
        Overlap prevents sentences from being cut in half at chunk boundaries.
        Recommendation: 10-20% of chunk_size (e.g., 100 chars for size=500).

    strategy : str
        "fixed"     - Split every chunk_size characters (simple, fast)
        "sentence"  - Split on sentence boundaries (better quality)
        "paragraph" - Split on paragraph breaks (good for structured text)
    """

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 40,
        strategy: str = "sentence",
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into a list of Chunks."""
        if not doc.text.strip():
            return []

        if self.strategy == "fixed":
            texts = self._fixed_split(doc.text)
        elif self.strategy == "sentence":
            texts = self._sentence_split(doc.text)
        elif self.strategy == "paragraph":
            texts = self._paragraph_split(doc.text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        chunks = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            # Inherit parent metadata and add chunk-specific fields.
            # This metadata travels with the embedding into the vector DB,
            # enabling later filtering by source file, chunk position, etc.
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": i,
                "chunk_count_in_doc": len(texts),
                "chunk_size": len(text),
                "chunk_strategy": self.strategy,
            }

            chunks.append(
                Chunk(
                    text=text,
                    metadata=chunk_metadata,
                    doc_id=doc.doc_id,
                    chunk_index=i,
                )
            )
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Convenience wrapper to chunk a list of documents."""
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(
                f"  Chunked '{doc.doc_id}': "
                f"{len(doc.text):,} chars → {len(chunks)} chunks"
            )
        return all_chunks

    # Splitting strategies

    def _fixed_split(self, text: str) -> list[str]:
        """
        Naive fixed-size split.

        Simplest strategy: every chunk_size characters, with overlap.
        Problem: often cuts in the middle of a word or sentence.
        When to use: quick prototyping, or when text has no natural structure.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            # Advance by (chunk_size - overlap) so chunks overlap
            start += self.chunk_size - self.overlap
        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        """
        Split on sentence boundaries then group into chunk_size windows.

        Better quality than fixed split: chunks always start/end at sentence
        boundaries, preserving grammatical coherence.

        Algorithm:
          1. Split text into individual sentences
          2. Greedily add sentences to the current chunk until chunk_size is reached
          3. When full, start a new chunk but backtrack overlap chars from the
             previous chunk (by including the tail sentences)

        Trade-off: slightly more complex, but retrieval quality improves noticeably
        because chunks contain complete thoughts.
        """
        import re

        # Split on period/exclamation/question followed by whitespace or end
        # This is a simple heuristic; for production use nltk.sent_tokenize
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return self._group_into_chunks(sentences)

    def _paragraph_split(self, text: str) -> list[str]:
        """
        Split on blank lines (paragraph breaks) then group into chunk_size windows.

        Best for: structured documents with clear paragraph breaks (essays,
        articles, documentation). Works poorly on continuous prose with no breaks.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return self._group_into_chunks(paragraphs)

    def _group_into_chunks(self, units: list[str]) -> list[str]:
        """
        Groups text units (sentences or paragraphs) into chunks of target size,
        with overlap by reusing the tail of the previous chunk.
        """
        chunks = []
        current_chunk_units = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)

            # If adding this unit would exceed chunk_size and we already have
            # content, finalize the current chunk and start a new one
            if current_len + unit_len > self.chunk_size and current_chunk_units:
                chunks.append(" ".join(current_chunk_units))

                # Overlap: keep units from the tail of the last chunk
                # until we've accumulated `overlap` characters
                overlap_units = []
                overlap_len = 0
                for prev_unit in reversed(current_chunk_units):
                    if overlap_len + len(prev_unit) > self.overlap:
                        break
                    overlap_units.insert(0, prev_unit)
                    overlap_len += len(prev_unit) + 1  # +1 for space

                current_chunk_units = overlap_units
                current_len = overlap_len

            current_chunk_units.append(unit)
            current_len += unit_len + 1  # +1 for space separator

        # Don't forget the last chunk
        if current_chunk_units:
            chunks.append(" ".join(current_chunk_units))

        return chunks



# Chunking strategy comparison (useful for experimentation)  


def compare_strategies(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Utility to compare all three strategies on the same text.
    Call this to understand how each strategy affects chunk count and quality.
    """
    results = {}
    for strategy in ("fixed", "sentence", "paragraph"):
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy)
        doc = Document(text=text, metadata={"source": "comparison_test"})
        chunks = chunker.chunk_document(doc)
        results[strategy] = {
            "count": len(chunks),
            "avg_size": sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
            "chunks": chunks,
        }
        print(
            f"  {strategy:12s}: {results[strategy]['count']} chunks, "
            f"avg {results[strategy]['avg_size']:.0f} chars each"
        )
    return results
