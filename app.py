from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort
import os
import math
import datetime

# Constants
ONNX_MODEL_PATH = "yolov8x.onnx"
YOLO_MODEL_URL = "https://github.com/botenstein/jetspotter-ai/releases/download/v1.0/yolov8x.onnx"
IMAGE_SIZE = 640
NUM_TILES = 5
AIRPLANE_CLASS_ID = 4
CONF_THRESHOLD = 0.3

# Download ONNX model if not present
def download_onnx_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        print("Downloading YOLOv8x ONNX model...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(ONNX_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ ONNX model downloaded.")
        else:
            raise Exception(f"Failed to download model: HTTP {response.status_code}")

download_onnx_model()

# Load Google Maps API key
with open('GoogleMapAPIKey.txt', 'r') as f:
    GOOGLE_MAPS_API_KEY = f.read().strip()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load ONNX model session
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Image preprocessing
import cv2
def preprocess_image(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.transpose(2, 0, 1) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# Inference with ONNX model
def run_inference(image):
    inputs = {ort_session.get_inputs()[0].name: preprocess_image(image)}
    outputs = ort_session.run(None, inputs)
    return outputs

# Serve frontend
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Detection endpoint
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        north, south, east, west = data['north'], data['south'], data['east'], data['west']
        lat_span = north - south
        lng_span = east - west
        lat_step = lat_span / NUM_TILES
        lng_step = lng_span / NUM_TILES

        stitched_image = Image.new('RGB', (IMAGE_SIZE * NUM_TILES, IMAGE_SIZE * NUM_TILES))

        for i in range(NUM_TILES):
            for j in range(NUM_TILES):
                sub_north = north - i * lat_step
                sub_south = sub_north - lat_step
                sub_west = west + j * lng_step
                sub_east = sub_west + lng_step
                center_lat = (sub_north + sub_south) / 2
                center_lng = (sub_east + sub_west) / 2
                bbox_width = abs(sub_east - sub_west)
                zoom = min(21, max(0, math.floor(math.log2(360 * (IMAGE_SIZE / 256) / bbox_width))))
                tile_url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lng}"
                    f"&zoom={zoom}&size={IMAGE_SIZE}x{IMAGE_SIZE}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
                )
                response = requests.get(tile_url)
                if response.status_code != 200:
                    print(f"⚠️ Failed tile {i},{j}")
                    continue
                tile_img = Image.open(BytesIO(response.content)).convert('RGB')
                stitched_image.paste(tile_img, (j * IMAGE_SIZE, i * IMAGE_SIZE))

        # Save image
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        stitched_filename = f'stitched_{timestamp}.jpg'
        stitched_image.save(stitched_filename)

        # Inference
        detections = run_inference(stitched_image)

        all_detections = []
        boxes = detections[0] if isinstance(detections, list) else detections
        for box in boxes[0]:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == AIRPLANE_CLASS_ID and conf > CONF_THRESHOLD:
                lat1 = north - (y1 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lat2 = north - (y2 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lng1 = west + (x1 / (IMAGE_SIZE * NUM_TILES)) * lng_span
                lng2 = west + (x2 / (IMAGE_SIZE * NUM_TILES)) * lng_span
                all_detections.append({
                    'class_id': int(cls),
                    'confidence': float(conf),
                    'lat1': max(lat1, lat2),
                    'lat2': min(lat1, lat2),
                    'lng1': min(lng1, lng2),
                    'lng2': max(lng1, lng2)
                })

        return jsonify({'message': 'Detection complete', 'detections': all_detections})

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
