import io
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import torchvision.transforms.functional as F

CLASS_NAMES = {0:"__background__",1:"apple",2:"orange",3:"pear",4:"watermelon",5:"korean_melon",6:"lemon",7:"grape",8:"pineapple",9:"cantaloupe",10:"dragon_fruit",11:"durian"}
NUM_CLASSES = 12
CONFIDENCE_THRESHOLD = 0.5
MAX_SIZE = 640

app = FastAPI(title="Fruit Object Detection API", version="1.0.0")

def load_model(weights_path="fruit_detector_model.pth"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model("fruit_detector_model.pth")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")
    model = None

def resize_image(image, max_size=MAX_SIZE):
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return image

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/health")
def health():
    return {"service":"Fruit Object Detection API","status":"running","model_loaded":model is not None,"supported_fruits":list(CLASS_NAMES.values())[1:]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if file.content_type not in ["image/jpeg","image/png","image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload JPG or PNG.")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_w, original_h = image.size
        image = resize_image(image)
        resized_w, resized_h = image.size
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")
    image_tensor = F.to_tensor(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    pred = predictions[0]
    boxes = pred["boxes"].tolist()
    labels = pred["labels"].tolist()
    scores = pred["scores"].tolist()
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= CONFIDENCE_THRESHOLD:
            detections.append({"label":CLASS_NAMES.get(label,"unknown"),"confidence":round(score,4),"box":{"xmin":round(box[0]*scale_x,1),"ymin":round(box[1]*scale_y,1),"xmax":round(box[2]*scale_x,1),"ymax":round(box[3]*scale_y,1)}})
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return JSONResponse(content={"filename":file.filename,"image_size":{"width":original_w,"height":original_h},"detections_count":len(detections),"detections":detections})
