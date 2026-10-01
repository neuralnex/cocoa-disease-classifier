from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cacao_disease_model.pth"

IMAGE_SIZE = 224

id2label = {
    0: "black_pod_rot",
    1: "healthy",
    2: "pod_borer",
}

label_names = {
    "black_pod_rot": "Black Pod Rot",
    "healthy": "Healthy",
    "pod_borer": "Pod Borer",
}


# -------------------------
# Model
# -------------------------

model = mobilenet_v3_small(weights=None)

model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

model.eval()


# -------------------------
# Image preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# -------------------------
# Prediction
# -------------------------

def predict_image(image: Image.Image):

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, prediction = torch.max(
            probabilities,
            dim=0
        )

    class_id = prediction.item()
    predicted_class = id2label[class_id]

    probabilities_dict = {
        id2label[i]: round(
            probabilities[i].item(),
            4
        )
        for i in range(3)
    }

    return {
        "prediction": predicted_class,
        "disease": label_names[predicted_class],
        "confidence": round(
            confidence.item(),
            4
        ),
        "probabilities": probabilities_dict,
    }