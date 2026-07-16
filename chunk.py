from pathlib import Path
import hashlib
import sys
import json
import re
import html
import tiktoken
import argparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

parser = argparse.ArgumentParser(
    description="Process and chunk Azure DI markdown files."
)
parser.add_argument(
    "--input", type=Path, required=True, help="Path to the .raw.md file"
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to save chunks.json"
)
args = parser.parse_args()

MARKDOWN_PATH = args.input
OUTPUT_PATH = args.output

HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]

MIN_CHUNK_TOKENS = 50
TARGET_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100

encoding = tiktoken.encoding_for_model("text-embedding-3-large")

PAGE_MARKER_RE = re.compile(r"@@PAGE:(\d+)@@")
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)


def count_tokens(text):
    # strip page markers for the count so they don't impact RecursiveCharacterTextSplitter limits
    clean_text_for_counting = PAGE_MARKER_RE.sub("", text)
    return len(encoding.encode(clean_text_for_counting))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_into_tagged_pages(raw_text):
    raw_pages = re.split(r"<!--\s*PageBreak\s*-->", raw_text)
    tagged_pages = []

    # Strictly count pages sequentially
    for i, raw_page in enumerate(raw_pages):
        page_num = i + 1
        tagged_pages.append(f"@@PAGE:{page_num}@@\n{raw_page}")

    return "\n".join(tagged_pages)


def fix_table_separator(md_table):
    lines = md_table.strip("\n").split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("|") and "-" in next_line:
                    break

            n_cols = line.count("|") - 1
            sep = "|" + "|".join([" --- "] * n_cols) + "|"
            lines.insert(i + 1, sep)
            break

    return "\n".join(lines)


def clean_markdown(text):
    text = COMMENT_RE.sub("", text)
    soup = BeautifulSoup(text, "html.parser")

    for table in soup.find_all("table"):
        table_md = fix_table_separator(md(str(table)).strip())
        table.replace_with(f"\n\n{table_md}\n\n")

    for figure in soup.find_all("figure"):
        caption = figure.find("figcaption")
        value = "Figure: " + (
            caption.get_text(" ", strip=True)
            if caption
            else figure.get_text(" ", strip=True)
        )
        figure.replace_with(f"\n\n{value}\n\n")

    text = html.unescape(str(soup))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([^\n])\n(#+ )", r"\1\n\n\2", text)

    return text.strip()


def format_paths(paths):
    """Truncates massive accumulated header paths to save tokens."""
    if not paths:
        return ""
    if len(paths) > 3:
        return f"{paths[0]} | ... | {paths[-1]}"
    return " | ".join(paths)


def extract_and_strip_pages(text):
    pages = sorted(set(int(p) for p in PAGE_MARKER_RE.findall(text)))
    clean_text = PAGE_MARKER_RE.sub("", text).strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, pages


def page_range_str(pages):
    if not pages:
        return None
    return str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"


if not MARKDOWN_PATH.exists():
    print(f"Error: input file not found: {MARKDOWN_PATH}")
    sys.exit(1)

raw = MARKDOWN_PATH.read_text(encoding="utf-8")
tagged = split_into_tagged_pages(raw)
markdown = clean_markdown(tagged)

print(f"Pages detected: {len(PAGE_MARKER_RE.findall(markdown))}")

# semantic split
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS, strip_headers=True
)
sections = header_splitter.split_text(markdown)
print(f"Sections: {len(sections)}")

# merge semantic chunks
merged = []
current_text = ""
current_paths = []
current_tokens = 0

for section in sections:
    tokens = count_tokens(section.page_content)

    # calculate the hierarchy path for this specific section
    section_path = " > ".join(
        section.metadata[level]
        for level in ("h1", "h2", "h3", "h4", "h5")
        if level in section.metadata
    )

    if (
        current_text
        and (current_tokens >= MIN_CHUNK_TOKENS)
        and (current_tokens + tokens > TARGET_CHUNK_TOKENS)
    ):
        merged.append(
            Document(
                page_content=current_text,
                metadata={"accumulated_path": format_paths(current_paths)},
            )
        )
        current_text = ""
        current_tokens = 0
        current_paths = []

    current_text += (
        ("\n\n" + section.page_content) if current_text else section.page_content
    )
    current_tokens += tokens

    if section_path and section_path not in current_paths:
        current_paths.append(section_path)

if current_text:
    if merged and current_tokens < MIN_CHUNK_TOKENS:
        prev = merged[-1]
        prev_paths = [p for p in prev.metadata["accumulated_path"].split(" | ") if p]
        merged[-1] = Document(
            page_content=prev.page_content + "\n\n" + current_text,
            metadata={
                "accumulated_path": " | ".join(
                    dict.fromkeys(prev_paths + current_paths)
                )
            },
        )
    else:
        merged.append(
            Document(
                page_content=current_text,
                metadata={"accumulated_path": format_paths(current_paths)},
            )
        )

print(f"Semantic chunks: {len(merged)}")

# fallback recursive split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=OVERLAP_TOKENS,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", " "],
)

final_docs = []
for doc in merged:
    if count_tokens(doc.page_content) <= MAX_CHUNK_TOKENS:
        final_docs.append(doc)
    else:
        sub_chunks = splitter.split_documents([doc])
        for i, sub_doc in enumerate(sub_chunks):
            if count_tokens(sub_doc.page_content) < MIN_CHUNK_TOKENS and final_docs:
                prev_doc = final_docs[-1]
                final_docs[-1] = Document(
                    page_content=prev_doc.page_content
                    + "\n\n"
                    + sub_doc.page_content.strip(),
                    metadata=prev_doc.metadata,
                )
            else:
                final_docs.append(sub_doc)

print(f"Final chunks: {len(final_docs)}")

chunks = []
current_page_memory = []

for i, doc in enumerate(final_docs):
    metadata = dict(doc.metadata)

    section_path = metadata.get("accumulated_path", "")

    body, pages = extract_and_strip_pages(doc.page_content)

    if pages:
        current_page_memory = pages
    elif not pages and current_page_memory:
        pages = [current_page_memory[-1]]

    text = f"{section_path}\n\n{body.strip()}".strip()

    metadata.update(
        {
            "doc_id": MARKDOWN_PATH.stem,
            "section_path": section_path,
            "pages": pages,
            "page_range": page_range_str(pages),
            "tokens": count_tokens(text),
            "length": len(text),
            "hash": sha256(text),
        }
    )

    if "accumulated_path" in metadata:
        del metadata["accumulated_path"]

    chunks.append(
        {
            "id": f"{MARKDOWN_PATH.stem}_{i}",
            "text": text,
            "metadata": metadata,
        }
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
)

if not chunks:
    print(f"Warning: no chunks produced from {MARKDOWN_PATH.name} (empty document?).")
    sys.exit(0)

counts = [c["metadata"]["tokens"] for c in chunks]

print(f"\nStats:")
print(f"  Min tokens: {min(counts)}")
print(f"  Median tokens: {sorted(counts)[len(counts)//2]}")
print(f"  Mean tokens: {sum(counts)/len(counts):.1f}")
print(f"  Max tokens: {max(counts)}")
print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH.name}\n")

print("Sample chunk page ranges:")
for c in chunks[:8]:
    preview_path = c["metadata"]["section_path"]
    if len(preview_path) > 55:
        preview_path = preview_path[:52] + "..."

    print(
        f"  {c['id']}: pages={c['metadata']['pages']} range={c['metadata']['page_range']} section={preview_path}"
    )
    # print(c["text"])
