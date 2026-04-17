import re
import os
import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.domain.entities.document_chunk import DocumentChunk
from src.domain.interfaces.i_document_ingestion_service import IDocumentIngestionService

log = logging.getLogger(__name__)


class DoclingDocumentIngestionService(IDocumentIngestionService):
    def __init__(self, docs_path: str = "./docs") -> None:
        self._docs_path = Path(docs_path)
        self._converter = self._build_converter()
        self._pdf_batch_pages = int(os.getenv("PDF_PARSE_BATCH_PAGES", "12"))
        self._docling_max_pages = int(os.getenv("DOCLING_MAX_PDF_PAGES", "30"))
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _build_converter(self) -> DocumentConverter:
        ocr_langs = [lang.strip() for lang in os.getenv("OCR_LANGS", "ru,en").split(",") if lang.strip()]
        use_gpu = os.getenv("OCR_USE_GPU", "false").lower() == "true"
        try:
            from docling_surya import SuryaOcrOptions

            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                ocr_model="suryaocr",
                allow_external_plugins=True,
                ocr_options=SuryaOcrOptions(lang=ocr_langs, use_gpu=use_gpu),
            )
            log.info("[Ingestion] Using Surya OCR backend: langs=%s use_gpu=%s", ocr_langs, use_gpu)
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
        except Exception as exc:
            raise RuntimeError(
                "Surya OCR is required but unavailable. Install and run with Python >=3.12 "
                "and ensure 'docling-surya' is installed."
            ) from exc

    def ingest(self, content: bytes, filename: str) -> list[DocumentChunk]:
        self._docs_path.mkdir(parents=True, exist_ok=True)
        save_path = self._docs_path / filename
        save_path.write_bytes(content)
        log.info("[Ingestion] Saved input document: path=%s", save_path)

        sections = self._parse_sections(save_path, filename)

        chunks: list[DocumentChunk] = []
        table_sections = 0
        for section in sections:
            cleaned = self._clean_noise(section)
            if len(cleaned) < 25:
                continue
            chunk_type = "table" if self._looks_like_table(cleaned) else "text"
            if chunk_type == "table":
                table_sections += 1
            for piece in self._splitter.split_text(cleaned):
                text = piece.strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        text=text,
                        source_file=filename,
                        page_number=None,
                        chunk_type=chunk_type,
                    )
                )
        if not chunks:
            raw_text = "\n".join(sections).strip()
            if raw_text:
                log.warning("[Ingestion] No chunks after strict cleaning, using relaxed chunking fallback")
                for piece in self._splitter.split_text(raw_text):
                    text = re.sub(r"\s+", " ", piece).strip()
                    if len(text) >= 20:
                        chunks.append(
                            DocumentChunk(
                                text=text,
                                source_file=filename,
                                page_number=None,
                                chunk_type="text",
                            )
                        )
        log.info(
            "[Ingestion] Chunking done: total_chunks=%s table_sections=%s text_sections=%s",
            len(chunks),
            table_sections,
            max(len(sections) - table_sections, 0),
        )
        return chunks

    def _parse_sections(self, save_path: Path, filename: str) -> list[str]:
        suffix = save_path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(save_path))
            total_pages = len(reader.pages)
            log.info("[Ingestion] PDF detected: pages=%s", total_pages)
        try:
            log.info("[Ingestion] Docling conversion started: file=%s", filename)
            conversion = self._converter.convert(str(save_path))
            markdown = conversion.document.export_to_markdown()
            sections = self._split_markdown_sections(markdown)
            log.info(
                "[Ingestion] Docling conversion done: markdown_len=%s sections=%s",
                len(markdown),
                len(sections),
            )
            if not sections:
                plain = re.sub(r"\s+", " ", markdown).strip()
                if plain:
                    log.info("[Ingestion] Structured section split empty, using raw markdown section")
                    sections = [plain]
            return sections
        except Exception as exc:
            # On large scanned 10-K files Docling OCR can fail with bad_alloc.
            # Keep indexing operational via lightweight batched fallback.
            log.warning("[Ingestion] Docling failed, using fallback parser: error=%s", exc)
            sections = self._fallback_sections(save_path)
            log.info("[Ingestion] Fallback parser done: sections=%s", len(sections))
            return sections

    def _fallback_sections(self, save_path: Path) -> list[str]:
        suffix = save_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            html = save_path.read_text(encoding="utf-8", errors="ignore")
            # crude but resilient html text fallback
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return [text] if text else []
        if suffix == ".pdf":
            reader = PdfReader(str(save_path))
            return self._fallback_pdf_sections_batched(reader)
        text = save_path.read_text(encoding="utf-8", errors="ignore").strip()
        return [text] if text else []

    def _fallback_pdf_sections_batched(self, reader: PdfReader) -> list[str]:
        pages: list[str] = []
        total_pages = len(reader.pages)
        log.info(
            "[Ingestion] Fallback PDF batch parsing started: total_pages=%s batch_pages=%s",
            total_pages,
            self._pdf_batch_pages,
        )
        for start in range(0, total_pages, self._pdf_batch_pages):
            end = min(start + self._pdf_batch_pages, total_pages)
            batch_idx = (start // self._pdf_batch_pages) + 1
            log.info("[Ingestion] Parsing PDF batch %s: pages=%s-%s", batch_idx, start + 1, end)
            for idx in range(start, end):
                extracted = (reader.pages[idx].extract_text() or "").strip()
                if extracted:
                    pages.append(f"# Page {idx + 1}\n{extracted}")
        log.info("[Ingestion] Fallback PDF batch parsing finished: extracted_pages=%s", len(pages))
        return pages

    @staticmethod
    def _split_markdown_sections(markdown: str) -> list[str]:
        parts = re.split(r"\n(?=#{1,6}\s)", markdown)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        pipe_lines = sum(1 for line in lines if "|" in line)
        return pipe_lines >= 2

    @staticmethod
    def _clean_noise(text: str) -> str:
        filtered_lines: list[str] = []
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if re.fullmatch(r"\d{1,4}", candidate):
                continue
            if re.search(r"page\s+\d+(\s+of\s+\d+)?", candidate.lower()):
                continue
            if len(candidate) < 4:
                continue
            filtered_lines.append(candidate)
        compact = " ".join(filtered_lines)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact
