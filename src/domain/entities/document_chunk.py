from dataclasses import dataclass


@dataclass(slots=True)
class DocumentChunk:
    text: str
    source_file: str
    page_number: int | None
    chunk_type: str
