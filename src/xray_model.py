import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/best_xray_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# -------------------------
# Build model architecture
# -------------------------
def build_model():
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    return model

# -------------------------
# Load trained model once
# -------------------------
_loaded_model = None

def load_xray_model():
    global _loaded_model

    if _loaded_model is not None:
        return _loaded_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    _loaded_model = build_model()
    _loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _loaded_model = _loaded_model.to(DEVICE)
    _loaded_model.eval()

    return _loaded_model

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Prediction function
# -------------------------
def predict_xray_condition(image_path):
    if not os.path.exists(image_path):
        return {
            "predicted_condition": "Unknown",
            "confidence": 0.0
        }

    try:
        model = load_xray_model()

        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        predicted_class = CLASS_NAMES[pred.item()]
        confidence_score = round(confidence.item() * 100, 2)

        return {
            "predicted_condition": predicted_class,
            "confidence": confidence_score
        }

    except Exception as e:
        print(f"[X-ray Model Error] {e}")
        return {
            "predicted_condition": "Unknown",
            "confidence": 0.0
        }