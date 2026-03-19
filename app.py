"""
EAI6010 Module 5 - Assignment 5
Fruit Object Detection Microservice

This FastAPI app exposes my Module 4 Faster R-CNN fruit detection model
as a REST API endpoint. It accepts an image upload and returns the
detected fruits with their bounding boxes and confidence scores.

Author: Eyinade Iyanuoluwa Joseph
Northeastern University, Vancouver
"""

import io
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torchvision.transforms.functional as F

# ----------------------------------------------------------------
# Class names — same 12 classes from Module 4
# Index 0 is background (reserved by Faster R-CNN), 1-11 are fruits
# ----------------------------------------------------------------
CLASS_NAMES = {
    0: "__background__",
    1: "apple",
    2: "orange",
    3: "pear",
    4: "watermelon",
    5: "korean_melon",
    6: "lemon",
    7: "grape",
    8: "pineapple",
    9: "cantaloupe",
    10: "dragon_fruit",
    11: "durian",
}

NUM_CLASSES = 12
CONFIDENCE_THRESHOLD = 0.5  # only return detections above 50% confidence

# ----------------------------------------------------------------
# App setup
# ----------------------------------------------------------------
app = FastAPI(
    title="Fruit Object Detection API",
    description=(
        "A Faster R-CNN model trained on the Kaggle Fruit Object Detection dataset. "
        "Upload an image to detect fruits and receive bounding box coordinates and confidence scores."
    ),
    version="1.0.0",
)

# ----------------------------------------------------------------
# Model loading
# We load the model once at startup so it doesn't reload on every request.
# This is important for performance on a free hosting tier.
# ----------------------------------------------------------------
def load_model(weights_path: str = "fruit_detector_model.pth"):
    """
    Recreate the same Faster R-CNN architecture from Module 4,
    then load our trained weights into it.
    """
    # Start with a pretrained Faster R-CNN backbone (same as Module 4)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classification head to match our 12 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Load our trained weights
    # map_location="cpu" ensures it works even if the server has no GPU
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()  # set to evaluation mode (disables dropout, batchnorm training behaviour)
    return model


# Load model at startup
try:
    model = load_model("fruit_detector_model.pth")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")
    model = None


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint — confirms the service is running."""
    return {
        "service": "Fruit Object Detection API",
        "status": "running",
        "usage": "POST an image to /predict to detect fruits",
        "supported_fruits": list(CLASS_NAMES.values())[1:],  # skip background
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a JPG or PNG image and returns detected fruits.

    Returns a list of detections, each with:
    - label: the fruit name (string)
    - confidence: model confidence score (0.0 to 1.0)
    - box: bounding box as [xmin, ymin, xmax, ymax] in pixels
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPG or PNG image."
        )

    # Read and convert the uploaded image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid JPG or PNG.")

    # Convert PIL image to a float tensor in [0, 1] range
    # Faster R-CNN expects a list of tensors, one per image
    image_tensor = F.to_tensor(image)

    # Run inference (no_grad saves memory since we're not training)
    with torch.no_grad():
        predictions = model([image_tensor])

    # predictions[0] contains boxes, labels, and scores for our single image
    pred = predictions[0]
    boxes = pred["boxes"].tolist()
    labels = pred["labels"].tolist()
    scores = pred["scores"].tolist()

    # Filter detections below the confidence threshold
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= CONFIDENCE_THRESHOLD:
            detections.append({
                "label": CLASS_NAMES.get(label, "unknown"),
                "confidence": round(score, 4),
                "box": {
                    "xmin": round(box[0], 1),
                    "ymin": round(box[1], 1),
                    "xmax": round(box[2], 1),
                    "ymax": round(box[3], 1),
                },
            })

    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return JSONResponse(content={
        "filename": file.filename,
        "image_size": {"width": image.width, "height": image.height},
        "detections_count": len(detections),
        "detections": detections,
    })
