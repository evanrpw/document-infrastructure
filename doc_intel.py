import argparse
import json
import logging
import os
import time
import sys
from pathlib import Path

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

client = DocumentIntelligenceClient(
    endpoint=os.environ["AZURE_DOC_INTEL_ENDPOINT"].rstrip("/"),
    credential=AzureKeyCredential(os.environ["AZURE_DOC_INTEL_KEY"]),
)


def analyze_with_retry(pdf_path: Path, max_attempts: int = 3):
    """Call Azure Document Intelligence with simple exponential backoff.

    Transient throttling/network errors are common at any real batch
    scale, so a single failed call shouldn't sink an otherwise-good file.
    """
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            with pdf_path.open("rb") as f:
                poller = client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=f,
                    output_content_format="markdown",
                )
            return poller.result()
        except Exception:
            if attempt == max_attempts:
                raise
            log.warning(
                f"  Attempt {attempt}/{max_attempts} failed for {pdf_path.name}, "
                f"retrying in {delay}s..."
            )
            time.sleep(delay)
            delay *= 2


def save_result(pdf_path: Path, output_root: Path):
    relative = pdf_path.relative_to(args.input)
    output_path = output_root / relative.parent / f"{pdf_path.stem}.docintel.json"

    if output_path.exists() and not args.force:
        log.info(f"Skipping {relative} (already processed)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Processing {relative}")

    result = analyze_with_retry(pdf_path)

    md_path = output_root / relative.parent / f"{pdf_path.stem}.raw.md"
    result_dict = result.as_dict()

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result_dict, fp, indent=2, ensure_ascii=False)

    md_path.write_text(result.content or "", encoding="utf-8")

    log.info(f"Saved JSON {output_path}")
    log.info(f"Saved Markdown {md_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--force", action="store_true", help="Reprocess files even if JSON exists"
)
args = parser.parse_args()

pdfs = sorted(args.input.rglob("*.pdf"))
log.info(f"Found {len(pdfs)} PDFs")

failures = []
for pdf in pdfs:
    try:
        save_result(pdf, args.output)
    except Exception:
        log.exception(f"Failed: {pdf}")
        failures.append(str(pdf))

if failures:
    log.error(f"{len(failures)} file(s) failed extraction: {failures}")
    sys.exit(1)
