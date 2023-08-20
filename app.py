from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageDraw, ImageFont  # Import ImageFont
import io
import torch
import tempfile
from fastapi.responses import FileResponse
import os

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read()))

    results = model(image)
    detected_objects = results.pred[0]

    object_info = []
    for det in detected_objects:
        label = model.names[int(det[5])]
        confidence = float(det[4])
        bounding_box = det[:4].tolist()
        object_info.append({"label": label, "confidence": confidence, "bounding_box": bounding_box})

    image_with_boxes = image.copy()
    for det in detected_objects:
        label = model.names[int(det[5])]
        confidence = float(det[4])
        bounding_box = det[:4].int().tolist()

        draw = ImageDraw.Draw(image_with_boxes)
        box_width = 10  
        font_size = 24  

        x_min, y_min, x_max, y_max = bounding_box
        draw.rectangle(bounding_box, outline="#7fff00", width=box_width)
        

        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text((x_min, y_min), f"{label} ({confidence:.2f})", fill="black", font=font)

    image_save_path = os.path.join(os.path.dirname(__file__), "image_with_boxes.jpg")
    image_with_boxes.save(image_save_path, format="JPEG")

    return {"object_info": object_info, "image_with_boxes_path": image_save_path}
