"""
rag/ingestion.py
----------------
Document Ingestion

What it does:
  Reads raw files from disk and converts them into a standardized Document
  structure that the rest of the pipeline can work with.

Design decisions:
  - We attach metadata (filename, file type, char count) at ingestion time
    so it can be stored alongside embeddings for later filtering.
  - PDFs require a third-party library (pypdf); we import it lazily so the
    system still works without it if you only use .txt files.
"""

import os
from importlib import import_module
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """
    The core unit of data flowing through the pipeline.

    text     : the raw string content of the document
    metadata : arbitrary key-value store for source info, dates, categories, etc.
               Stored alongside the embedding vectors so you can filter later.
    doc_id   : a stable identifier (defaults to filename or a generated UUID)
    """
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        # Assign a doc_id if none was given
        if self.doc_id is None:
            import uuid
            self.doc_id = str(uuid.uuid4())


class DocumentLoader:
    """
    Loads documents from a file or directory.

    Supported formats: .txt, .md, .pdf
    Adding a new format = adding one more branch in _read_file().
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def load_file(self, path: str) -> Document:
        """Load a single file into a Document."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        text = self._read_file(path, ext)

        # Attach source metadata — this is what enables metadata filtering later
        metadata = {
            "source": os.path.basename(path),
            "file_path": path,
            "file_type": ext.lstrip("."),
            "char_count": len(text),
        }

        return Document(text=text, metadata=metadata, doc_id=os.path.basename(path))

    def load_directory(self, directory: str) -> list[Document]:
        """
        Recursively load all supported files from a directory.

        Trade-off: Loading everything eagerly is simple but uses more memory
        for large corpora. For millions of docs you'd stream lazily instead.
        """
        docs = []
        for root, _, files in os.walk(directory):
            for fname in sorted(files):  # sorted = deterministic order
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    fpath = os.path.join(root, fname)
                    try:
                        doc = self.load_file(fpath)
                        docs.append(doc)
                        print(f"  Loaded: {fname} ({doc.metadata['char_count']:,} chars)")
                    except Exception as e:
                        print(f"  WARNING: Skipped {fname}: {e}")
        return docs

    def load_text(self, text: str, source_name: str = "user_input") -> Document:
        """
        Create a Document directly from a string.
        Useful for programmatic ingestion without touching the filesystem.
        """
        return Document(
            text=text,
            metadata={"source": source_name, "file_type": "text", "char_count": len(text)},
        )


    # Private helpers


    def _read_file(self, path: str, ext: str) -> str:
        if ext in (".txt", ".md"):
            return self._read_text(path)
        elif ext == ".pdf":
            return self._read_pdf(path)
        raise ValueError(f"No reader for extension: {ext}")

    def _read_text(self, path: str) -> str:
        # Try UTF-8 first; fall back to latin-1 for older files
        for encoding in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Could not decode {path} with any supported encoding")

    def _read_pdf(self, path: str) -> str:
        try:
            PdfReader = import_module("pypdf").PdfReader  # optional dependency
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF ingestion. "
                "Install it with: pip install pypdf"
            )
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)