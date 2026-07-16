import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient, UpdateOne
import warnings

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TesseractCliOcrOptions,
)
from docling.chunking import HybridChunker
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)

warnings.filterwarnings(
    "ignore", message="You appear to be connected to a CosmosDB cluster"
)
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

logging.getLogger("pymongo").setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="data/pdf")
parser.add_argument(
    "--skip_ingested", action="store_true", help="skip files already in DB"
)
parser.add_argument("--dry_run", action="store_true", help="simulate without DB write")
parser.add_argument("-v", "--verbose", action="store_true")

parser.add_argument("--source_type_depth", type=int, default=0)
parser.add_argument("--author_depth", type=int, default=1)
parser.add_argument("--fips_depth", type=int, default=-1)

parser.add_argument("--batch_size", type=int, default=8, help="embedding batch size")
parser.add_argument("--min_chunk_chars", type=int, default=40)
parser.add_argument("--max_junk_token_ratio", type=float, default=0.4)
parser.add_argument("--corruption_threshold", type=float, default=0.01)
parser.add_argument(
    "--chunk_corruption_threshold",
    type=float,
    default=0.03,
    help=(
        "Threshold for the post-chunking corruption sweep. Chunks carry more "
        "surrounding context than a single page-level average, so this is set "
        "a bit looser than --corruption_threshold to avoid over-flagging."
    ),
)
parser.add_argument("--force_ocr", action="store_true")
parser.add_argument("--disable_ocr_fallback", action="store_true")
parser.add_argument(
    "--disable_chunk_sweep",
    action="store_true",
    help="disable the post-chunking corruption sweep (page-level detection only)",
)
parser.add_argument(
    "--ocr_image_scale",
    type=float,
    default=2.0,
    help="rasterization scale used when force_ocr is active; higher = sharper OCR input",
)
args = parser.parse_args()

if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
    api_version=os.getenv("AZURE_EMBEDDING_VERSION"),
)
container = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))[os.getenv("DB_NAME")][
    os.getenv("DB_RAG_CONTAINER")
]

_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_PUNCTUATION_REGEX = r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]"


# --- Heuristics ---
def corruption_score(text: str) -> tuple[float, list[str]]:
    if not text:
        return 0.0, []

    if text.count("\ufffd") / len(text) > 0.01:
        return 1.0, ["High density of Unicode replacement characters ()"]

    tokens = text.split()
    if not tokens:
        return 0.0, []

    eval_tokens = [
        t for t in tokens if len(re.sub(_PUNCTUATION_REGEX, "", t)) >= 3 or t == "#l"
    ]
    if not eval_tokens:
        return 0.0, []

    bad = []
    for t in eval_tokens:
        core = re.sub(_PUNCTUATION_REGEX, "", t)
        if not core and t != "#l":
            continue

        # 1. Spatial/Background Glyph Splicing
        if re.search(r"[a-zA-Z]\d[a-zA-Z]", core) or re.search(
            r"\d[a-zA-Z]{2,}\d", core
        ):
            bad.append(t)
        elif re.search(r"[a-zA-Z][,_;][a-zA-Z]", t):
            bad.append(t)
        elif re.search(rf"[{_CONSONANTS}]{{5,}}", core.lower()):
            bad.append(t)
        # 2. Visual OCR Artifacts
        elif re.search(r"20[5-9]\d", core):
            bad.append(t)
        elif re.search(r"[a-zA-Z][\\|][a-zA-Z]", t):
            bad.append(t)
        elif t == "#l":
            bad.append(t)

    return len(bad) / len(eval_tokens), bad


def get_page_texts(doc) -> dict:
    pages: dict = {}
    for item in doc.texts:
        text = getattr(item, "text", "") or ""
        if not text:
            continue
        for prov in getattr(item, "prov", []):
            pages.setdefault(prov.page_no, []).append(text)
    return {p: "\n".join(chunks) for p, chunks in pages.items()}


def detect_corrupted_pages(doc, threshold: float) -> list:
    flagged = []
    for page_no, text in get_page_texts(doc).items():
        score, _ = corruption_score(text)
        if score > threshold:
            flagged.append((page_no, score))
    return sorted(flagged)


def is_junk_chunk(text: str, min_chars: int, max_junk_ratio: float) -> tuple[bool, str]:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True, f"too short ({len(stripped)} chars < {min_chars})"

    tokens = stripped.split()
    if not tokens:
        return True, "empty after strip"

    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    ratio = short_tokens / len(tokens)
    if ratio > max_junk_ratio:
        return True, f"{ratio:.0%} of tokens are <=2 chars"

    return False, ""


def save_chunks_to_file(chunk_records, file_path):
    output_path = Path(file_path).with_name(f"{Path(file_path).stem}_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)


def save_chunks_to_md(kept_chunks, file_path):
    output_path = Path(file_path).with_name(f"{Path(file_path).stem}_final_chunks.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Final Chunks for {Path(file_path).name}\n\n")
        f.write(f"Total embedded chunks: {len(kept_chunks)}\n\n---\n\n")
        for i, (chunk, text, used_ocr) in enumerate(kept_chunks):
            f.write(f"## Chunk {i+1} | OCR Used: {used_ocr}\n\n")
            f.write(f"{text}\n\n---\n\n")


def get_converter(force_ocr: bool):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True

    pipeline_options.enable_remote_services = True

    pipeline_options.do_chart_extraction = False
    pipeline_options.generate_picture_images = False
    pipeline_options.images_scale = args.ocr_image_scale if force_ocr else 1.0

    pipeline_options.do_picture_description = False
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url=os.getenv("AZURE_VISION_ENDPOINT"),
        headers={"api-key": os.getenv("AZURE_VISION_API_KEY")},
        model=os.getenv("AZURE_VISION_MODEL_NAME", "gpt-4o-mini"),
        prompt="You are an expert data extraction assistant. Describe this diagram in detail. If it contains a chart, graph, or infographic with numerical data, extract all of that data into a precise, structured markdown table.",
    )

    if force_ocr:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    else:
        pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.get_encoding("cl100k_base"),
    max_tokens=8191,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=2048,
    repeat_table_header=True,
    serializer_provider=MDTableSerializerProvider(),
)


def make_id(rel_path, chunk_text):
    return hashlib.sha256(f"{rel_path}\x00{chunk_text}".encode()).hexdigest()


def content_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()


def path_metadata(file_path):
    try:
        rel = Path(file_path).relative_to(args.base_dir)
        parts = rel.parts[:-1]
        source_type = (
            parts[args.source_type_depth]
            if args.source_type_depth < len(parts)
            else "general"
        )
        author = (
            parts[args.author_depth]
            if args.author_depth != -1 and args.author_depth < len(parts)
            else "Unknown"
        )
        fips = (
            parts[args.fips_depth]
            if args.fips_depth != -1 and args.fips_depth < len(parts)
            else None
        )
        return str(rel), source_type, author, fips
    except ValueError:
        return str(file_path), "general", "Unknown", None


def chunk_pages(chunk) -> set:
    return set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov)


def ocr_pages(file_path, pages) -> list:
    """Force-OCR and chunk a specific set of page numbers. Returns a list of
    (chunk, used_ocr=True) tuples. Shared by the initial page-level flagging
    pass and the post-chunking corruption sweep below."""
    pages = sorted(pages)
    if not pages:
        return []
    ocr_converter = get_converter(force_ocr=True)
    results = []
    for page_no in pages:
        try:
            page_doc = ocr_converter.convert(
                file_path, page_range=(page_no, page_no)
            ).document
            results.extend((c, True) for c in chunker.chunk(page_doc))
        except Exception as e:
            log.error(f"    OCR re-conversion of page {page_no} failed: {e}")
    return results


def embed_chunks(texts, max_retries=5):
    embeddings = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        for attempt in range(max_retries):
            try:
                resp = openai_client.embeddings.create(
                    model=os.getenv("AZURE_EMBEDDING_MODEL_NAME"), input=batch
                )
                embeddings.extend(item.embedding for item in resp.data)
                break
            except Exception as e:
                if "RateLimitError" in type(e).__name__ or (
                    hasattr(e, "status_code") and e.status_code == 429
                ):
                    wait = 15 * (attempt + 1)
                    log.warning(f"  Rate limited by Azure! Waiting {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    if attempt == max_retries - 1:
                        raise
                    wait = 2**attempt
                    log.warning(
                        f"  Embedding batch failed ({e}); retrying in {wait}s..."
                    )
                    time.sleep(wait)
    return embeddings


def ingest_file(file_path):
    log.info(f"Ingesting: {file_path}")
    rel_path, source_type, author, fips = path_metadata(file_path)

    fast_converter = get_converter(force_ocr=False)
    try:
        fast_doc = fast_converter.convert(file_path).document
    except Exception as e:
        log.error(f"  Extraction failed: {e}")
        return False

    out_docling = Path(file_path).with_name(f"{Path(file_path).stem}_docling.md")
    out_docling.write_text(fast_doc.export_to_markdown(), encoding="utf-8")

    if args.disable_ocr_fallback:
        flagged = []
    elif args.force_ocr:
        flagged = [(p, 1.0) for p in get_page_texts(fast_doc)]
    else:
        flagged = detect_corrupted_pages(fast_doc, args.corruption_threshold)

    flagged_pages = {p for p, _ in flagged}
    total_pages = len(get_page_texts(fast_doc))

    log.info(
        f"PDF Parsing Summary: {len(fast_doc.texts)} text blocks, {len(fast_doc.tables)} tables, {len(flagged_pages)} page(s) flagged for OCR."
    )

    chunk_bundle = []

    if flagged_pages and total_pages > 0 and (len(flagged_pages) / total_pages) > 0.20:
        log.warning(f"  >20% of pages corrupted. Re-OCR'ing entire document.")
        try:
            ocr_converter = get_converter(force_ocr=True)
            ocr_doc = ocr_converter.convert(file_path).document
            chunk_bundle = [(c, True) for c in chunker.chunk(ocr_doc)]
        except Exception as e:
            log.error(f"  Full-document OCR failed: {e}")
            return False
    else:
        native_chunks = [
            c for c in chunker.chunk(fast_doc) if not (chunk_pages(c) & flagged_pages)
        ]
        chunk_bundle = [(c, False) for c in native_chunks]

        if flagged_pages:
            pages_str = ", ".join(f"p{p} ({s:.1%})" for p, s in flagged)
            log.warning(f"  Re-OCR'ing flagged page(s) only: {pages_str}")
            chunk_bundle.extend(ocr_pages(file_path, flagged_pages))

        # Chunk-level corruption sweep
        if (
            not args.disable_chunk_sweep
            and not args.force_ocr
            and not args.disable_ocr_fallback
        ):
            extra_pages = set()
            for c in native_chunks:
                if chunk_pages(c) & flagged_pages:
                    continue
                score, _ = corruption_score(c.text)
                if score > args.chunk_corruption_threshold:
                    extra_pages |= chunk_pages(c)

            if extra_pages:
                log.warning(
                    f"  Chunk-level sweep caught {len(extra_pages)} additional "
                    f"corrupted page(s) missed by page-level detection: "
                    f"{sorted(extra_pages)}"
                )
                chunk_bundle = [
                    (c, used)
                    for c, used in chunk_bundle
                    if used or not (chunk_pages(c) & extra_pages)
                ]
                chunk_bundle.extend(ocr_pages(file_path, extra_pages))
                flagged_pages |= extra_pages

    if not chunk_bundle:
        log.warning("  Skipping: no chunks produced.")
        return False

    chunk_records = []
    kept = []
    n_junk = 0
    for i, (chunk, used_ocr) in enumerate(chunk_bundle):
        contextualized = chunker.contextualize(chunk)
        junk, reason = is_junk_chunk(
            chunk.text, args.min_chunk_chars, args.max_junk_token_ratio
        )

        record = {
            "index": i,
            "text": chunk.text,
            "contextualized_text": contextualized,
            "metadata": chunk.meta.model_dump(),
            "pages": sorted(chunk_pages(chunk)),
            "is_junk": junk,
            "junk_reason": reason,
            "used_ocr": used_ocr,
        }
        chunk_records.append(record)
        if junk:
            n_junk += 1
        else:
            kept.append((chunk, contextualized, used_ocr))

    save_chunks_to_file(chunk_records, file_path)
    save_chunks_to_md(kept, file_path)

    log.info(
        f"  -> {len(chunk_bundle)} chunks produced "
        f"({sum(1 for _, u in chunk_bundle if u)} from OCR), "
        f"{n_junk} filtered as junk, {len(kept)} kept for embedding."
    )

    if not kept:
        log.warning("  Skipping: no chunks left after junk filtering.")
        return False

    if args.dry_run:
        log.info(f"  [DRY RUN] {len(kept)} chunks would be embedded.")
        return True

    # db
    texts = [t for _, t, _ in kept]
    embeddings = embed_chunks(texts)

    ops = []
    for i, ((chunk, text, used_ocr), embedding) in enumerate(zip(kept, embeddings)):
        doc_entry = {
            "_id": make_id(rel_path, text),
            "chunk": text,
            "contentHash": content_hash(text),
            "embedding": embedding,
            "sourceFile": str(file_path),
            "source": author,
            "sourceType": source_type,
            "chunkIndex": i,
            "metadata": chunk.meta.to_dict(),
            "usedOcr": used_ocr,
            "pageNumbers": sorted(chunk_pages(chunk)),
        }
        if fips is not None:
            doc_entry["fips"] = fips

        ops.append(
            UpdateOne({"_id": doc_entry["_id"]}, {"$set": doc_entry}, upsert=True)
        )

    try:
        container.bulk_write(ops, ordered=False)
        log.info(f"  Ingested {len(ops)} chunks.")
    except Exception as e:
        log.error(f"  MongoDB write failed: {e}. Rolling back {len(ops)} chunks...")
        try:
            container.delete_many({"sourceFile": str(file_path)})
            log.info(f"  Rollback complete.")
        except Exception as re:
            log.error(f"  Rollback also failed: {re}")
        return False

    return True


def main():
    ingested = set(container.distinct("sourceFile")) if args.skip_ingested else set()
    if ingested:
        log.info(f"Found {len(ingested)} already-ingested files; will skip.")

    files = [
        Path(r) / f
        for r, _, fs in os.walk(args.base_dir)
        for f in fs
        if f.lower().endswith(".pdf")
    ]

    total = skipped = succeeded = failed = 0
    for f in files:
        total += 1
        if str(f) in ingested:
            log.info(f"Skipping (already ingested): {f}")
            skipped += 1
            continue
        if ingest_file(str(f)):
            succeeded += 1
        else:
            failed += 1

    log.info(
        f"\nDone. {total} total | {skipped} skipped | {succeeded} ingested | {failed} failed"
    )


if __name__ == "__main__":
    main()
