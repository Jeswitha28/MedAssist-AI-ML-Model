import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =========================
# CONFIG
# =========================
TRAIN_DIR = "data/chest_xray/train"
TEST_DIR = "data/chest_xray/test"
MODEL_SAVE_PATH = "models/best_xray_model.pth"

BATCH_SIZE = 8          # reduce to 4 if laptop is slow
IMG_SIZE = 224
EPOCHS = 5              # increase later to 8-10 if needed
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
# CHECK PATHS
# =========================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

os.makedirs("models", exist_ok=True)

# =========================
# SET SEED
# =========================
torch.manual_seed(SEED)
random.seed(SEED)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# LOAD FULL TRAIN DATASET
# =========================
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

# We need separate transform for validation
full_train_dataset_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)

class_names = full_train_dataset.classes
print("Classes:", class_names)   # ['NORMAL', 'PNEUMONIA']

# =========================
# TRAIN / VAL SPLIT
# =========================
dataset_size = len(full_train_dataset)
val_size = int(dataset_size * VAL_SPLIT)
train_size = dataset_size - val_size

train_indices, val_indices = random_split(
    range(dataset_size),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Create subsets with different transforms
train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
val_subset = torch.utils.data.Subset(full_train_dataset_eval, val_indices.indices)

# Test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Testing samples: {len(test_dataset)}")

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

# Only train final layer
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# =========================
# TRAINING LOOP
# =========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved!")

print("\n🎉 Training completed!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# =========================
# LOAD BEST MODEL FOR TESTING
# =========================
print("\n========== Testing Best Model ==========")

best_model = models.resnet18(weights=None)
num_features = best_model.fc.in_features
best_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
best_model = best_model.to(DEVICE)
best_model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total

print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")