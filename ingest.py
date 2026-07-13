"""
Filename: ingest.py
Author: Evan Paces-Wiles
Created: 2026-06-10
Description: Process, chunk, embed (with Azure OpenAI), and ingest PDF and text files into MongoDB (Azure Document DB) for RAG applications.
"""

# TODO: better handling for tables, columns, and malformed pdfs; semantic chunking with pdf structure (so tables are one chunk)

import argparse
import hashlib
import logging
import os
import re
from pathlib import Path

import pymupdf4llm
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient, UpdateOne
import warnings

warnings.filterwarnings(
    "ignore", message="You appear to be connected to a CosmosDB cluster"
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress verbose httpx logs
log = logging.getLogger(__name__)

# Args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    default="data/pdf",
    help="root folder to ingest and base for metadata path parsing",
)
parser.add_argument("--file_type", default="both", choices=["pdf", "txt", "both"])
parser.add_argument(
    "--source_type_depth",
    type=int,
    default=0,
    help="folder depth for source_type (0-indexed, excluding filename)",
)
parser.add_argument(
    "--author_depth",
    type=int,
    default=1,
    help="folder depth for author (-1 to disable)",
)
parser.add_argument(
    "--fips_depth",
    type=int,
    default=-1,
    help="folder depth for fips code (-1 to disable)",
)
parser.add_argument("--separators", nargs="+", default=["\n\n", "\n", ". ", " "])
parser.add_argument("--chunk_size", type=int, default=500)
parser.add_argument("--overlap", type=int, default=100)
parser.add_argument(
    "--no_page_numbers", action="store_true", help="disable page number tracking"
)
parser.add_argument(
    "--no_chunking",
    action="store_true",
    help="disable chunking and embed the entire file as a single document",
)
parser.add_argument(
    "--embedding_model", default=os.getenv("AZURE_EMBEDDING_MODEL_NAME")
)
parser.add_argument("--azure_endpoint", default=os.getenv("AZURE_EMBEDDING_ENDPOINT"))
parser.add_argument("--azure_api_key", default=os.getenv("AZURE_FOUNDRY_API_KEY"))
parser.add_argument("--azure_api_version", default=os.getenv("AZURE_FOUNDRY_VERSION"))
parser.add_argument(
    "--mongo_connection_string", default=os.getenv("MONGO_CONNECTION_STRING")
)
parser.add_argument("--db_name", default=os.getenv("DB_NAME"))
parser.add_argument("--db_container", default=os.getenv("DB_RAG_CONTAINER"))
parser.add_argument(
    "--skip_ingested", action="store_true", help="skip files already in DB"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="embedding batch size (set lower if embedding model rate limited)",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="simulate processing and chunking without embedding or writing to the database",
)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

# Clients

openai_client = AzureOpenAI(
    azure_endpoint=args.azure_endpoint,
    api_key=args.azure_api_key,
    api_version=args.azure_api_version,
)
container = MongoClient(args.mongo_connection_string)[args.db_name][args.db_container]

# Extraction

_REGEX_CLEANUPS = [
    (r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", " "),
    (r"(?<=[a-zA-Z,])\. (?=[a-z])", " "),
    (r",\.", ", "),
    (r"\.{2,}", ". "),
]


def _clean(text):
    for pattern, repl in _REGEX_CLEANUPS:
        text = re.sub(pattern, repl, text)
    return text


def extract_pages(file_path):
    """Return list of {'text': str, 'metadata': {'page_number': int|None}}."""
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error(f"  Error reading {file_path}: {e}")
            return None
        return [{"text": text, "metadata": {"page_number": None}}]
    elif ext == ".pdf":
        try:
            pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            for p in pages:
                if p.get("text"):
                    p["text"] = _clean(p["text"])
            return pages
        except Exception as e:
            log.error(f"  Error extracting {file_path}: {e}")
            return None
    else:
        log.error(f"  Unsupported file type: {ext}")
        return None


# Chunking


# simple version based on langchain's RecursiveCharacterTextSplitter, plus start/end char offsets for page number mapping
def chunk_text(text):
    """Yield (chunk_text, start_char, end_char) tuples with overlap."""

    if args.no_chunking:
        yield text.strip(), 0, len(text)
        return

    def split(text, seps):
        if not text.strip():
            return []
        sep, rest = seps[0], seps[1:]
        parts, chunks, current = text.split(sep), [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate.split()) <= args.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())

                if len(part.split()) > args.chunk_size and rest:
                    chunks.extend(split(part, rest))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())
        return chunks

    raw = split(text, args.separators)
    pos = 0
    for i, chunk in enumerate(raw):
        if i > 0:
            chunk = " ".join(raw[i - 1].split()[-args.overlap :]) + " " + chunk
        start = text.find(raw[i][:50], pos)
        if start == -1:
            start = pos
        end = start + len(chunk)
        pos = start + 1
        yield chunk.strip(), start, end


# Helpers


def make_id(rel_path, chunk_text):
    return hashlib.sha256(f"{rel_path}\x00{chunk_text}".encode()).hexdigest()


# _id is hash of (relative file path + chunk text, store content hash of chunk text for debugging/deduplication
def content_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()


def path_metadata(file_path):
    """Return (rel_path_str, source_type, author, fips) from folder structure."""
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
        # ADD FIPS EXTRACTION:
        fips = (
            parts[args.fips_depth]
            if args.fips_depth != -1 and args.fips_depth < len(parts)
            else None
        )
        return str(rel), source_type, author, fips
    except ValueError:
        return str(file_path), "general", "Unknown", None


def chunk_pages(chunk_start, chunk_end, page_map):
    if args.no_page_numbers:
        return None
    return [
        n for s, e, n in page_map if chunk_start < e and chunk_end > s and n is not None
    ]


def embed_chunks(texts):
    embeddings = []
    for i in range(0, len(texts), args.batch_size):
        resp = openai_client.embeddings.create(
            model=args.embedding_model, input=texts[i : i + args.batch_size]
        )
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


# Ingestion


def ingest_file(file_path):
    log.info(f"Ingesting: {file_path}")
    pages = extract_pages(file_path)
    if not pages:
        log.warning(f"  Skipping: no data extracted.")
        return False

    rel_path, source_type, author, fips = path_metadata(file_path)
    if author == "Unknown":
        internal = pages[0].get("metadata", {}).get("author", "").strip()
        if internal:
            author = internal

    full_text, page_map = "", []
    for page in pages:
        text = page.get("text", "")
        p_num = page.get("metadata", {}).get("page_number")
        start = len(full_text)
        full_text += text + "\n\n"
        page_map.append((start, len(full_text), p_num))

    chunks = list(chunk_text(full_text))
    if not chunks:
        log.warning(f"  Skipping: no chunks produced.")
        return False

    log.info(f"  {len(chunks)} chunks across {len(pages)} pages.")

    # ADD THIS DRY RUN BLOCK:
    if args.dry_run:
        log.info("  [DRY RUN] Skipping embedding and database upload.")
        # Optionally print a preview of the first chunk's metadata for verification
        if chunks:
            preview_chunk = chunks[0][0][:100].replace("\n", " ")
            log.info(
                f"  [DRY RUN PREVIEW] source={source_type} | fips={fips} | text={preview_chunk}..."
            )
        return True

    log.info("  Embedding...")
    try:
        embeddings = embed_chunks([c for c, *_ in chunks])
    except Exception as e:
        log.error(f"  Embedding failed: {e}")
        return False

    ops = []
    for i, ((chunk, c_start, c_end), embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            "_id": make_id(rel_path, chunk),
            "chunk": chunk,
            "contentHash": content_hash(chunk),
            "embedding": embedding,
            "sourceFile": file_path,
            "source": author,
            "sourceType": source_type,
            "chunkIndex": i,
        }

        if fips is not None:
            doc["fips"] = fips

        pages_hit = chunk_pages(c_start, c_end, page_map)
        if pages_hit is not None:
            doc["pageNumbers"] = pages_hit
        log.debug(f"  [{i}] pages={pages_hit} | {chunk[:20]}...")
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

    try:
        container.bulk_write(ops, ordered=False)
    except Exception as e:
        log.error(f"  MongoDB write failed: {e}. Rolling back {len(ops)} chunks...")
        try:
            container.delete_many({"sourceFile": file_path})
            log.info(f"  Rollback complete.")
        except Exception as re:
            log.error(f"  Rollback also failed: {re}")
        return False

    log.info(f"  Done.")
    return True


# Main

VALID_EXTS = {"pdf": {".pdf"}, "txt": {".txt"}, "both": {".pdf", ".txt"}}


def main():
    ingested = set(container.distinct("sourceFile")) if args.skip_ingested else set()
    if ingested:
        log.info(f"Found {len(ingested)} already-ingested files; will skip.")

    files = [
        Path(r) / f
        for r, _, fs in os.walk(args.base_dir)
        for f in fs
        if Path(f).suffix.lower() in VALID_EXTS[args.file_type]
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
            log.warning(f"Failed to ingest: {f}")

    log.info(
        f"\nDone. {total} found | {skipped} skipped | {succeeded} ingested | {failed} failed"
    )


if __name__ == "__main__":
    main()
