from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageDraw, ImageFont
import io
import torch
from fastapi.responses import StreamingResponse
import os
import base64

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

@app.get("/")
async def root():
    return "Dentisto Api 2"

@app.get("/check")
async def root():
    return "Calling Dentisto Api 2 for every 14 min...."

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
        draw.text((x_min, y_min), f"{label} ({confidence:.2f})", fill="black")

    # Save the image with boxes as a base64-encoded string
    image_buffer = io.BytesIO()
    image_with_boxes.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    image_base64 = base64.b64encode(image_buffer.read()).decode("utf-8")

    # Create a JSON response that includes labels and the base64-encoded image
    response_data = {
        "labels": object_info,
        "image_base64": image_base64
    }
    print(response_data)

    return response_data























# from fastapi import FastAPI, UploadFile, File
# from PIL import Image, ImageDraw, ImageFont
# import io
# import torch
# from fastapi.responses import StreamingResponse
# import os

# app = FastAPI()

# model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
# @app.get("/")
# async def root():
#     return "Dentisto Api 2"


# @app.get("/check")
# async def root():
#     return "Calling Dentisto Api 2 for every 14 min...."
# @app.post("/detect/")
# async def detect_objects(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(file.file.read()))

#     results = model(image)
#     detected_objects = results.pred[0]

#     object_info = []
#     for det in detected_objects:
#         label = model.names[int(det[5])]
#         confidence = float(det[4])
#         bounding_box = det[:4].tolist()
#         object_info.append({"label": label, "confidence": confidence, "bounding_box": bounding_box})

#     image_with_boxes = image.copy()
#     for det in detected_objects:
#         label = model.names[int(det[5])]
#         confidence = float(det[4])
#         bounding_box = det[:4].int().tolist()

#         draw = ImageDraw.Draw(image_with_boxes)
#         box_width = 10  
#         font_size = 24  

#         x_min, y_min, x_max, y_max = bounding_box
#         draw.rectangle(bounding_box, outline="#7fff00", width=box_width)
#         draw.text((x_min, y_min), f"{label} ({confidence:.2f})", fill="black")

#     image_buffer = io.BytesIO()
#     image_with_boxes.save(image_buffer, format="JPEG")
#     image_buffer.seek(0)  # Reset the buffer pointer to the beginning

#     return StreamingResponse(
#         content=image_buffer,
#         media_type="image/jpeg",
#     )
