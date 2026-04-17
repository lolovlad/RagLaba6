import argparse
from pathlib import Path

from config_logging import setup_logging
from src.application.services.document_indexing_service import DocumentIndexingService
from src.container import container


def index_file(file_path: Path) -> int:
    service = container.resolve(DocumentIndexingService)
    content = file_path.read_bytes()
    return service.index_document(content=content, filename=file_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index 10-K PDF/HTML files from docs folder")
    parser.add_argument("--docs-path", default="./docs", help="Folder with PDF/HTML files")
    args = parser.parse_args()

    setup_logging()
    docs_path = Path(args.docs_path)
    files = [p for p in docs_path.glob("*") if p.suffix.lower() in {".pdf", ".html", ".htm"}]
    if not files:
        print("No PDF/HTML files found in docs directory")
        return
    total_chunks = 0
    for file_path in files:
        indexed = index_file(file_path)
        total_chunks += indexed
        print(f"Indexed {indexed} chunks from {file_path.name}")
    print(f"Done. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()
