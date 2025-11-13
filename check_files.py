from pathlib import Path
import os
import sys

def main(dataset_path):
    print(f"Current working directory: {os.getcwd()}")
    dataset_dir = Path(dataset_path)
    print(f"Looking for data in folder: {dataset_dir.resolve()}")
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} does not exist")
        sys.exit(1)
    edf_files_lower = list(dataset_dir.glob('*.edf'))
    edf_files_upper = list(dataset_dir.glob('*.EDF'))
    all_files = edf_files_lower + edf_files_upper
    print(f"Found {len(all_files)} EDF files total:")
    for file in all_files:
        print(f" - {file} (Exists: {file.exists()}) Size: {file.stat().st_size} bytes")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_files.py <dataset_dir>")
        sys.exit(1)
    main(sys.argv[1])
