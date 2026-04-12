import os
import zipfile
import shutil
import subprocess
import sys

# =========================
# CONFIG
# =========================
DATA_DIR = "data"
ZIP_FILE = os.path.join(DATA_DIR, "chest-xray-pneumonia.zip")
EXTRACT_DIR = DATA_DIR
FINAL_DATASET_DIR = os.path.join(DATA_DIR, "chest_xray")

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

# =========================
# CREATE DATA FOLDER
# =========================
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# CHECK IF ALREADY EXISTS
# =========================
if os.path.exists(FINAL_DATASET_DIR):
    print(f"✅ Dataset already exists at: {FINAL_DATASET_DIR}")
    print("If you want to re-download, delete the folder first.")
    sys.exit()

# =========================
# DOWNLOAD DATASET
# =========================
print("⬇️ Downloading dataset from Kaggle...")

try:
    subprocess.run(
        [
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
    print("1. You installed kaggle: pip install kaggle")
    print("2. kaggle.json is placed at:")
    print("   C:\\Users\\mouni\\.kaggle\\kaggle.json")
    print("3. Your internet is working")
    sys.exit()

# =========================
# VERIFY ZIP FILE
# =========================
if not os.path.exists(ZIP_FILE):
    print(f"\n❌ ZIP file not found: {ZIP_FILE}")
    print("Check if Kaggle downloaded with a different filename.")
    sys.exit()

# =========================
# EXTRACT ZIP
# =========================
print("📦 Extracting dataset...")

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("✅ Extraction completed!")

# =========================
# CLEANUP ZIP
# =========================
try:
    os.remove(ZIP_FILE)
    print("🗑️ Removed ZIP file after extraction.")
except:
    pass

# =========================
# VERIFY FINAL STRUCTURE
# =========================
expected_train = os.path.join(FINAL_DATASET_DIR, "train")
expected_val = os.path.join(FINAL_DATASET_DIR, "val")
expected_test = os.path.join(FINAL_DATASET_DIR, "test")

if all(os.path.exists(p) for p in [expected_train, expected_val, expected_test]):
    print("\n🎉 Dataset is ready!")
    print(f"📁 Final dataset path: {FINAL_DATASET_DIR}")
    print("\nExpected structure:")
    print("data/chest_xray/")
    print("├── train/")
    print("├── val/")
    print("└── test/")
else:
    print("\n⚠️ Extraction completed, but expected folders not found.")
    print("Please check the extracted files manually inside 'data/'")