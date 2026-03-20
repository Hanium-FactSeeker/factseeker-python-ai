"""High-level article fact-check pipeline.

This package provides an article-focused fact checking flow analogous to the
YouTube flow, without modifying existing modules. It reuses the vector search,
FAISS, and fact-check chains, and introduces a dedicated article claim extractor.
"""

