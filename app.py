import gradio as gr
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.serialization import add_safe_globals

# Allow YOLOv8 loading on HF Spaces
add_safe_globals([DetectionModel])

# Load YOLOv8 nano (COCO pretrained)
model = YOLO("yolov8n.pt")

def detect(img):
    results = model(img)
    result_img = results[0].plot()
    return result_img

iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Output"),
    title="Aerial Object Detector",
    description="Upload an image to detect objects using YOLOv8 Nano.",
)

iface.launch()
