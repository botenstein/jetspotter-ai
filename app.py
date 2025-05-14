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
ONNX_MODEL_PATH = "yolov8s.onnx"
YOLO_MODEL_URL = "https://github.com/botenstein/jetspotter-ai/releases/download/v1.0/yolov8s.onnx"
IMAGE_SIZE = 640
NUM_TILES = 1
AIRPLANE_CLASS_ID = 4
CONF_THRESHOLD = 0.1

# Download ONNX model if missing
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        print("⬇️ Downloading YOLOv8s ONNX model...")
        r = requests.get(YOLO_MODEL_URL, stream=True)
        if r.status_code == 200:
            with open(ONNX_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print("✅ Download complete.")
        else:
            raise Exception(f"Model download failed: HTTP {r.status_code}")

download_model()

# Load Google Maps API key
with open('GoogleMapAPIKey.txt', 'r') as f:
    GOOGLE_MAPS_API_KEY = f.read().strip()

# Flask app
app = Flask(__name__)
CORS(app)

# Load ONNX model
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Preprocess image for ONNX
def preprocess(img: Image.Image):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    x = np.array(img).astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(x, axis=0)

# Run model
def infer(img: Image.Image):
    x = preprocess(img)
    return ort_session.run(None, {ort_session.get_inputs()[0].name: x})

# Post-process (model must have NMS built-in)
def extract_boxes(preds):
    # Expecting output: (1, num_detections, 6) = [x1, y1, x2, y2, conf, class_id]
    boxes = []
    for box in preds[0][0]:  # remove batch dim
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf >= CONF_THRESHOLD and int(cls) == AIRPLANE_CLASS_ID:
            boxes.append([x1, y1, x2, y2, conf, int(cls)])
    return boxes

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        north, south, east, west = data['north'], data['south'], data['east'], data['west']
        lat_span, lng_span = north - south, east - west
        lat_step, lng_step = lat_span / NUM_TILES, lng_span / NUM_TILES

        stitched = Image.new("RGB", (IMAGE_SIZE * NUM_TILES, IMAGE_SIZE * NUM_TILES))

        for i in range(NUM_TILES):
            for j in range(NUM_TILES):
                n = north - i * lat_step
                s = n - lat_step
                w = west + j * lng_step
                e = w + lng_step
                lat, lng = (n + s) / 2, (e + w) / 2
                zoom = min(21, max(0, math.floor(math.log2(360 * (IMAGE_SIZE / 256) / abs(e - w)))))
                url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}"
                    f"&zoom={zoom}&size={IMAGE_SIZE}x{IMAGE_SIZE}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
                )
                resp = requests.get(url)
                if resp.status_code != 200:
                    print(f"⚠️ Failed tile {i},{j}")
                    continue
                tile = Image.open(BytesIO(resp.content)).convert("RGB")
                stitched.paste(tile, (j * IMAGE_SIZE, i * IMAGE_SIZE))

        # Save debug image
        name = f"stitched_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        stitched.save(name)
        print(f"✅ Stitched image saved as {name}")

        # Inference + boxes
        raw_preds = infer(stitched)
        boxes = extract_boxes(raw_preds)

        # Convert pixel to lat/lng
        results = []
        for x1, y1, x2, y2, conf, cls in boxes:
            lat1 = north - (y1 / (IMAGE_SIZE * NUM_TILES)) * lat_span
            lat2 = north - (y2 / (IMAGE_SIZE * NUM_TILES)) * lat_span
            lng1 = west + (x1 / (IMAGE_SIZE * NUM_TILES)) * lng_span
            lng2 = west + (x2 / (IMAGE_SIZE * NUM_TILES)) * lng_span
            results.append({
                'class_id': cls,
                'confidence': conf,
                'lat1': max(lat1, lat2),
                'lat2': min(lat1, lat2),
                'lng1': min(lng1, lng2),
                'lng2': max(lng1, lng2)
            })

        return jsonify({'message': 'Detection complete', 'detections': results})

    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
