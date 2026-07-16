import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
import sys
import subprocess

load_dotenv()

parser = argparse.ArgumentParser(description="Run the ingest pipeline.")
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="Simulate chunking and metadata creation without embedding or uploading.",
)
parser.add_argument(
    "--file",
    type=str,
    help="Relative path to a single file to process (e.g., Academic/Marwah et al. - 2010.pdf)",
)
parser.add_argument(
    "--sample_run",
    action="store_true",
    help="Process exactly one file from each top-level folder category for testing.",
)
args = parser.parse_args()

BASE_DIR = Path("data/pdf")
OUTPUT_DIR = Path("data/out")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# skip full azure scan if single file or sample run
if not args.file and not args.sample_run:
    print("--- Step 1: Syncing Azure Document Intelligence Extractions (PDFs only) ---")
    extraction = subprocess.run(
        [
            "python",
            "doc_intel.py",
            "--input",
            str(BASE_DIR),
            "--output",
            str(OUTPUT_DIR),
        ],
        check=False,
    )
    if extraction.returncode != 0:
        print(
            "Warning: extraction reported failures for one or more files "
            "(see log above). Continuing with whatever succeeded; failed "
            "files will be skipped below since their markdown won't exist."
        )
else:
    print("--- Testing/Single File Mode Active: Skipping bulk Azure folder scan ---")

# db check
if not args.dry_run:
    container = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))[os.getenv("DB_NAME")][
        os.getenv("DB_RAG_CONTAINER")
    ]

    db_files = list(container.find({}, {"sourceFile": 1, "sourceHash": 1}))

    ingested_map = {doc["sourceFile"]: doc.get("sourceHash") for doc in db_files}
else:
    ingested_map = {}
    print("\n[DRY RUN ENABLED] Skipping database checks and upload simulation mode on.")

print("\n--- Step 2 & 3: Chunking and Uploading ---")

if args.file:
    all_files = [BASE_DIR / args.file]
    if not all_files[0].exists():
        print(f"Error: Could not find {all_files[0]}")
        exit(1)
elif args.sample_run:
    # Gather one file from each top-level category folder
    print("[SAMPLE RUN ACTIVE] Filtering file targets to one per category...")
    raw_candidates = list(BASE_DIR.rglob("*.pdf")) + list(BASE_DIR.rglob("*.txt"))

    category_samples = {}
    for source_file in raw_candidates:
        relative_path = source_file.relative_to(BASE_DIR)
        top_folder = (
            relative_path.parts[0] if len(relative_path.parts) > 0 else "general"
        )

        if top_folder not in category_samples:
            category_samples[top_folder] = source_file

    all_files = list(category_samples.values())
    print(f"Found categories: {list(category_samples.keys())}")
else:
    # target everything
    all_files = list(BASE_DIR.rglob("*.pdf")) + list(BASE_DIR.rglob("*.txt"))

failures = []

for source_file in all_files:
    local_hash = file_sha256(source_file)

    if (
        str(source_file) in ingested_map
        and ingested_map[str(source_file)] == local_hash
    ):
        print(f"Skipping {source_file.name} (already in DB and unchanged).")
        continue

    print(f"\nProcessing chunks & upload for: {source_file.name}")

    relative_path = source_file.relative_to(BASE_DIR)
    raw_md_path = OUTPUT_DIR / relative_path.parent / f"{source_file.stem}.raw.md"
    chunks_json_path = (
        OUTPUT_DIR / relative_path.parent / f"{source_file.stem}_chunks.json"
    )

    # bypass Azure for .txt files
    if source_file.suffix.lower() == ".txt":
        raw_md_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_file, raw_md_path)
    else:
        # pdf: verify doc_intel created md file
        if not raw_md_path.exists():
            print(
                f"Warning: Expected markdown file missing at {raw_md_path}. Skipping."
            )
            continue

    try:
        subprocess.run(
            [
                "python",
                "chunk.py",
                "--input",
                str(raw_md_path),
                "--output",
                str(chunks_json_path),
            ],
            check=True,
        )
        # dynamic metadata based on dir structure
        top_folder = (
            relative_path.parts[0] if len(relative_path.parts) > 0 else "general"
        )

        author_depth = "1"
        fips_depth = "-1"

        if top_folder == "PublicComment":
            author_depth = "-1"
            fips_depth = "1"
        elif top_folder == "Academic" or top_folder == "Government":
            author_depth = "-1"
            fips_depth = "-1"

        upload_cmd = [
            "python",
            "upload.py",
            "--input_json",
            str(chunks_json_path),
            "--original_pdf",
            str(source_file),
            "--base_dir",
            str(BASE_DIR),
            "--author_depth",
            author_depth,
            "--fips_depth",
            fips_depth,
        ]

        if args.dry_run:
            upload_cmd.append("--dry_run")

        subprocess.run(upload_cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error processing {source_file.name}. Moving to next file.")
        failures.append(str(source_file))
        continue

if failures:
    print(f"\nPipeline finished with {len(failures)} failures:")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)

print("\n--- Pipeline Complete ---")
