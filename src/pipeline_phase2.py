"""Phase 2 pipeline for cleaning, validating, and exporting historical data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.cleaning import clean_entity, deduplicate_entities
from src.loader import ENTITIES_PATH, PROCESSED_DIR, load_json, save_json
from src.validation import validate_entities


OUTPUT_JSON_PATH = PROCESSED_DIR / "historical_entities_clean.json"
OUTPUT_CSV_PATH = PROCESSED_DIR / "historical_entities_clean.csv"
OUTPUT_REPORT_PATH = PROCESSED_DIR / "validation_report.json"

CSV_COLUMNS = [
    "id",
    "nombre",
    "tipo",
    "epoca",
    "ubicacion",
    "resumen",
    "descripcion_larga",
    "importancia",
    "lugares_relacionados",
    "personajes_relacionados",
    "eventos_relacionados",
    "etiquetas",
    "anio_inicio",
    "anio_fin",
    "fuente_base",
]

LIST_COLUMNS = {
    "lugares_relacionados",
    "personajes_relacionados",
    "eventos_relacionados",
    "etiquetas",
}


def export_clean_csv(entities: list[dict[str, Any]], path: str | Path) -> Path:
    """Export cleaned valid entities to CSV for manual inspection."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for entity in entities:
            row: dict[str, Any] = {}
            for column in CSV_COLUMNS:
                value = entity.get(column)
                if column in LIST_COLUMNS:
                    row[column] = " | ".join(value or [])
                elif value is None:
                    row[column] = ""
                else:
                    row[column] = value
            writer.writerow(row)

    return resolved_path


def run_pipeline() -> dict[str, Any]:
    """Run the complete Phase 2 cleaning and validation pipeline."""
    raw_entities = load_json(ENTITIES_PATH)
    cleaned_entities = [clean_entity(entity) for entity in raw_entities]
    deduplicated_entities, duplicates = deduplicate_entities(cleaned_entities)
    report = validate_entities(deduplicated_entities)

    valid_entities = [
        entity
        for entity, entity_report in zip(deduplicated_entities, report["entities"])
        if entity_report["valid"]
    ]

    report["summary"]["duplicate_ids_removed"] = len(duplicates["duplicate_ids"])
    report["summary"]["duplicate_names_removed"] = len(duplicates["duplicate_names"])
    report["duplicates"] = duplicates

    save_json(OUTPUT_JSON_PATH, valid_entities)
    export_clean_csv(valid_entities, OUTPUT_CSV_PATH)
    save_json(OUTPUT_REPORT_PATH, report)

    return {
        "raw_entities": len(raw_entities),
        "cleaned_entities": len(cleaned_entities),
        "deduplicated_entities": len(deduplicated_entities),
        "valid_entities": len(valid_entities),
        "invalid_entities": report["summary"]["invalid_entities"],
        "entities_with_warnings": report["summary"]["entities_with_warnings"],
        "duplicate_ids_removed": report["summary"]["duplicate_ids_removed"],
        "duplicate_names_removed": report["summary"]["duplicate_names_removed"],
        "json_output": str(OUTPUT_JSON_PATH),
        "csv_output": str(OUTPUT_CSV_PATH),
        "report_output": str(OUTPUT_REPORT_PATH),
    }


def main() -> None:
    """Execute the Phase 2 pipeline and print a concise summary."""
    summary = run_pipeline()

    print("Phase 2 pipeline completed successfully.")
    print(f"Raw entities: {summary['raw_entities']}")
    print(f"Cleaned entities: {summary['cleaned_entities']}")
    print(f"Deduplicated entities: {summary['deduplicated_entities']}")
    print(f"Valid entities exported: {summary['valid_entities']}")
    print(f"Invalid entities: {summary['invalid_entities']}")
    print(f"Entities with warnings: {summary['entities_with_warnings']}")
    print(f"Duplicate IDs removed: {summary['duplicate_ids_removed']}")
    print(f"Duplicate names removed: {summary['duplicate_names_removed']}")
    print(f"Clean JSON: {summary['json_output']}")
    print(f"Clean CSV: {summary['csv_output']}")
    print(f"Validation report: {summary['report_output']}")


if __name__ == "__main__":
    main()
