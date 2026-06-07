"""
OccGait Dataset Download Script.
Downloads and prepares the OccGait dataset from BNU-IVC GitHub repository.
"""
import os
import sys
import subprocess
import argparse


def download_occgait(output_dir: str = "data/OccGait") -> None:
    """Download OccGait dataset from GitHub.
    
    The dataset is hosted at: https://github.com/BNU-IVC/OccGait
    
    Args:
        output_dir: Directory to save the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    repo_url = "https://github.com/BNU-IVC/OccGait.git"
    
    print("=" * 60)
    print("OccGait Dataset Download")
    print("=" * 60)
    print(f"Repository: {repo_url}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Check if already downloaded
    if os.listdir(output_dir):
        print(f"[INFO] Directory {output_dir} is not empty.")
        print("[INFO] Skipping download. Delete the directory to re-download.")
        return
    
    # Clone the repository
    print("[1/3] Cloning OccGait repository...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print("[OK] Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git clone failed: {e.stderr}")
        print("[TIP] Make sure git is installed and you have internet access.")
        print("[TIP] You can also manually download from: https://github.com/BNU-IVC/OccGait")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] git command not found.")
        print("[TIP] Install git or manually download the dataset.")
        sys.exit(1)
    
    # Verify download
    print("[2/3] Verifying dataset structure...")
    expected_items = ['README.md']
    for item in expected_items:
        path = os.path.join(output_dir, item)
        if os.path.exists(path):
            print(f"  [OK] Found: {item}")
        else:
            print(f"  [WARN] Missing: {item}")
    
    # Print summary
    print("[3/3] Dataset summary:")
    total_files = sum(len(files) for _, _, files in os.walk(output_dir))
    total_dirs = sum(1 for _, dirs, _ in os.walk(output_dir)) - 1
    print(f"  Directories: {total_dirs}")
    print(f"  Files: {total_files}")
    print()
    print("[DONE] OccGait dataset is ready!")
    print()
    print("IMPORTANT: The OccGait dataset may require additional data files")
    print("to be downloaded separately following the instructions in the")
    print("repository's README.md. Please check:")
    print(f"  {os.path.join(output_dir, 'README.md')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OccGait dataset")
    parser.add_argument("--output", type=str, default="data/OccGait",
                       help="Output directory")
    args = parser.parse_args()
    
    download_occgait(args.output)
