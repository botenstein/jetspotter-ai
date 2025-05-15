import os
import math
import logging
import datetime
from io import BytesIO

import numpy as np
from PIL import Image
import requests
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------- Configuration ----------------------
ONNX_MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s.onnx")
YOLO_MODEL_URL = os.getenv("MODEL_URL", "https://github.com/botenstein/jetspotter-ai/releases/download/v1.0/yolov8s.onnx")

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))
NUM_TILES = int(os.getenv("NUM_TILES", 1))  # future-proofing
AIRPLANE_CLASS_ID = int(os.getenv("AIRPLANE_CLASS_ID", 4))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.1))
SAVE_STITCHED_IMAGE = bool(int(os.getenv("SAVE_STITCHED_IMAGE", 1)))

# ---------------------- Logging Setup ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------- Model Download ----------------------
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        logging.info("⬇️ Downloading ONNX model...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(ONNX_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            logging.info("✅ Model downloaded successfully.")
        else:
            raise RuntimeError(f"Model download failed: HTTP {response.status_code}")

download_model()
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# ---------------------- Load Google API Key ----------------------
try:
    with open('GoogleMapAPIKey.txt', 'r') as f:
        GOOGLE_MAPS_API_KEY = f.read().strip()
except Exception as e:
    raise RuntimeError(f"Failed to read Google Maps API key: {e}")

# ---------------------- Flask Setup ----------------------
app = Flask(__name__)
CORS(app)

# ---------------------- Image Processing ----------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    x = np.array(img).astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(x, axis=0)

def run_inference(img: Image.Image) -> np.ndarray:
    x = preprocess_image(img)
    return ort_session.run(None, {ort_session.get_inputs()[0].name: x})

def extract_detections(predictions) -> list:
    boxes = []
    for box in predictions[0][0]:  # shape: (N, 6)
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf >= CONF_THRESHOLD and int(cls) == AIRPLANE_CLASS_ID:
            boxes.append([x1, y1, x2, y2, conf, int(cls)])
    return boxes

# ---------------------- Routes ----------------------
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        north, south = data['north'], data['south']
        east, west = data['east'], data['west']
        lat_span = north - south
        lng_span = east - west
        lat_step = lat_span / NUM_TILES
        lng_step = lng_span / NUM_TILES

        stitched_img = Image.new("RGB", (IMAGE_SIZE * NUM_TILES, IMAGE_SIZE * NUM_TILES))
        filename = None

        for i in range(NUM_TILES):
            for j in range(NUM_TILES):
                n = north - i * lat_step
                s = n - lat_step
                w = west + j * lng_step
                e = w + lng_step
                lat, lng = (n + s) / 2, (e + w) / 2

                zoom = min(21, max(0, math.floor(math.log2(360 * (IMAGE_SIZE / 256) / abs(e - w)))))
                tile_url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?"
                    f"center={lat},{lng}&zoom={zoom}&size={IMAGE_SIZE}x{IMAGE_SIZE}"
                    f"&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
                )

                resp = requests.get(tile_url)
                if resp.status_code != 200:
                    logging.warning(f"Failed to fetch tile ({i}, {j}) - HTTP {resp.status_code}")
                    continue

                tile = Image.open(BytesIO(resp.content)).convert("RGB")
                stitched_img.paste(tile, (j * IMAGE_SIZE, i * IMAGE_SIZE))

        # Save stitched image (optional)
        if SAVE_STITCHED_IMAGE:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stitched_{timestamp}.jpg"
            stitched_img.save(filename)
            logging.info(f"🖼 Stitched image saved: {filename}")

        # Run detection
        predictions = run_inference(stitched_img)
        boxes = extract_detections(predictions)

        results = []
        for x1, y1, x2, y2, conf, cls in boxes:
            lat1 = north - (y1 / (IMAGE_SIZE * NUM_TILES)) * lat_span
            lat2 = north - (y2 / (IMAGE_SIZE * NUM_TILES)) * lat_span
            lng1 = west + (x1 / (IMAGE_SIZE * NUM_TILES)) * lng_span
            lng2 = west + (x2 / (IMAGE_SIZE * NUM_TILES)) * lng_span
            results.append({
                "class_id": cls,
                "confidence": conf,
                "lat1": max(lat1, lat2),
                "lat2": min(lat1, lat2),
                "lng1": min(lng1, lng2),
                "lng2": max(lng1, lng2)
            })

        logging.info(f"✅ Detection complete: {len(results)} aircraft(s) found")

        # Clean up stitched image if saved
        if filename and os.path.exists(filename):
            os.remove(filename)
            logging.info(f"🧹 Deleted temporary file: {filename}")

        return jsonify({
            "message": "Detection complete",
            "detections": results
        })

    except Exception as e:
        logging.exception("❌ Error during detection")
        return jsonify({"error": str(e)}), 500

# ---------------------- Run Server ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
