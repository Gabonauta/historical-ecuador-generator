#!/usr/bin/env python3
"""CLI entrypoint to build the local RAG index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_historical_entities
from src.rag_indexer import build_and_save_index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index construction."""
    parser = argparse.ArgumentParser(
        description="Construye el indice RAG local para historical-ecuador-generator."
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Proveedor de embeddings para construir el indice local.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Modelo de embeddings opcional. Si no se indica, se usa el default del provider.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the local index from the historical JSON dataset."""
    args = parse_args()
    entities = load_historical_entities()
    result = build_and_save_index(
        entities,
        embedding_provider=args.embedding_provider,
        model=args.embedding_model,
    )

    print("Indice RAG construido correctamente.")
    print(f"Chunks: {result['chunk_count']}")
    print(f"Entidades: {result['entity_count']}")
    print(f"Dimension: {result['embedding_dimension']}")
    print(f"Provider embeddings: {result['embedding_provider']}")
    print(f"Modelo embeddings: {result['embedding_model']}")
    print(f"Chunks guardados en: {result['chunks_path']}")
    print(f"Embeddings guardados en: {result['embeddings_path']}")
    print(f"Metadata guardada en: {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
