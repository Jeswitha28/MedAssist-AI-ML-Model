import os
import zipfile
import subprocess
import sys

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # MedAssist/

# =========================
# CONFIG
# =========================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ZIP_FILE = os.path.join(DATA_DIR, "chest-xray-pneumonia.zip")
FINAL_DATASET_DIR = os.path.join(DATA_DIR, "chest_xray")
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(FINAL_DATASET_DIR):
    print(f"✅ Dataset already exists at: {FINAL_DATASET_DIR}")
    print("If you want to re-download, delete the folder first.")
    sys.exit()

print("⬇️ Downloading dataset from Kaggle...")

try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            DATA_DIR
        ],
        check=True
    )
    print("✅ Download completed!")
except subprocess.CalledProcessError:
    print("\n❌ Kaggle download failed.")
    print("Please make sure:")
    print("1. pip install kaggle")
    print("2. kaggle.json exists at:")
    print("   C:\\Users\\mouni\\.kaggle\\kaggle.json")
    sys.exit()

if not os.path.exists(ZIP_FILE):
    print(f"\n❌ ZIP file not found: {ZIP_FILE}")
    sys.exit()

print("📦 Extracting dataset...")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("✅ Extraction completed!")

try:
    os.remove(ZIP_FILE)
    print("🗑️ Removed ZIP file.")
except:
    pass

expected_train = os.path.join(FINAL_DATASET_DIR, "train")
expected_val = os.path.join(FINAL_DATASET_DIR, "val")
expected_test = os.path.join(FINAL_DATASET_DIR, "test")

if all(os.path.exists(p) for p in [expected_train, expected_val, expected_test]):
    print("\n🎉 Dataset is ready!")
    print(f"📁 Final dataset path: {FINAL_DATASET_DIR}")
else:
    print("\n⚠️ Extraction completed, but expected folders not found.")
    print("Please check manually inside data/")