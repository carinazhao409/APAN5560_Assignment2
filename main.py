from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

import io
import torch
import torchvision.transforms as transforms

from cnn_model import CNN

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cifar10_cnn.pth"


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="CNN image classifier trained on the CIFAR-10 dataset",
    version="1.0"
)


# --------------------------------------------------
# 2. Choose device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 3. CIFAR-10 class names
# --------------------------------------------------

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


# --------------------------------------------------
# 4. Load trained CNN model
# --------------------------------------------------

model = CNN().to(device)

model.load_state_dict(
    torch.load(
    MODEL_PATH,
    map_location=device
)
)

model.eval()


# --------------------------------------------------
# 5. Define image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "CIFAR-10 Image Classifier API",
        "endpoint": "/predict"
    }


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_data = await file.read()

        # Convert uploaded file to RGB image
        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        # Apply the same preprocessing used during training
        image = transform(image)

        # Add batch dimension:
        # [3, 64, 64] -> [1, 3, 64, 64]
        image = image.unsqueeze(0)

        # Move image to device
        image = image.to(device)

        # Make prediction
        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        predicted_class = classes[
            predicted.item()
        ]

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(
                confidence.item(),
                4
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(e)}"
        )