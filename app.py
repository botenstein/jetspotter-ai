from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
import math
import datetime
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# === Config ===
GOOGLE_MAPS_API_KEY = 'YOUR_API_KEY_HERE'  # Replace this
YOLO_MODEL_PATH = 'yolov8x.pt'
YOLO_MODEL_URL = 'https://github.com/botenstein/jetspotter-ai/releases/download/v1.0/yolov8x.pt'
IMAGE_SIZE = 640
NUM_TILES = 5
AIRPLANE_CLASS_IDS = [4]
CONF_THRESHOLD = 0.3

# === Download model if needed ===
def download_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        print("🔄 Downloading yolov8x.pt from GitHub...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        with open(YOLO_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ YOLOv8x model downloaded.")
    else:
        print("📦 Model already exists.")

download_yolo_model()
model = YOLO(YOLO_MODEL_PATH)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

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
                zoom = max(0, min(21, math.floor(math.log2(360 * (IMAGE_SIZE / 256) / bbox_width))))

                tile_url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?"
                    f"center={center_lat},{center_lng}"
                    f"&zoom={zoom}&size={IMAGE_SIZE}x{IMAGE_SIZE}"
                    f"&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
                )

                response = requests.get(tile_url)
                if response.status_code != 200:
                    print(f"⚠️ Failed tile {i},{j}")
                    continue

                tile_img = Image.open(BytesIO(response.content)).convert("RGB")
                stitched_image.paste(tile_img, (j * IMAGE_SIZE, i * IMAGE_SIZE))

        # Save stitched image for debug/logs
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        stitched_filename = f'stitched_{timestamp}.jpg'
        stitched_image.save(stitched_filename)
        print(f"📸 Saved: {stitched_filename}")

        # Run YOLO inference
        results = model(stitched_image)
        detections = []

        if results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls not in AIRPLANE_CLASS_IDS or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lat1 = north - (y1 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lat2 = north - (y2 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lng1 = west + (x1 / (IMAGE_SIZE * NUM_TILES)) * lng_span
                lng2 = west + (x2 / (IMAGE_SIZE * NUM_TILES)) * lng_span

                detections.append({
                    'class_id': cls,
                    'confidence': conf,
                    'lat1': max(lat1, lat2),
                    'lat2': min(lat1, lat2),
                    'lng1': min(lng1, lng2),
                    'lng2': max(lng1, lng2)
                })

        return jsonify({
            'message': 'Detection complete',
            'detections': detections
        })

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({ 'error': str(e) }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
