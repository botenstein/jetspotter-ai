from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
import math
import datetime
from ultralytics import YOLO
import os


YOLO_MODEL_PATH = "yolov8x.pt"
YOLO_MODEL_URL = "https://github.com/botenstein/jetspotter-ai/releases/download/v1.0/yolov8x.pt"

def download_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Downloading YOLOv8x model from GitHub...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(YOLO_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("YOLOv8x model downloaded successfully.")
        else:
            print(f"Failed to download model. HTTP status code: {response.status_code}")
            exit(1)

# Call this before model loading
download_yolo_model()


app = Flask(__name__)
CORS(app)

with open('GoogleMapAPIKey.txt', 'r') as f:
    GOOGLE_MAPS_API_KEY = f.read().strip()

model = YOLO(YOLO_MODEL_PATH)

IMAGE_SIZE = 640
NUM_TILES = 5  # 3x3 grid → final stitched image 1920x1920
AIRPLANE_CLASS_IDS = [4]
CONF_THRESHOLD = 0.3

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        north, south, east, west = data['north'], data['south'], data['east'], data['west']

        lat_span = north - south
        lng_span = east - west
        lat_step = lat_span / NUM_TILES
        lng_step = lng_span / NUM_TILES

        # Create empty stitched image
        stitched_image = Image.new('RGB', (IMAGE_SIZE * NUM_TILES, IMAGE_SIZE * NUM_TILES))

        # Download and paste tiles into stitched image
        for i in range(NUM_TILES):
            for j in range(NUM_TILES):
                sub_north = north - i * lat_step
                sub_south = sub_north - lat_step
                sub_west = west + j * lng_step
                sub_east = sub_west + lng_step

                center_lat = (sub_north + sub_south) / 2
                center_lng = (sub_east + sub_west) / 2
                bbox_width = abs(sub_east - sub_west)
                zoom = math.floor(math.log2(360 * (IMAGE_SIZE / 256) / bbox_width))
                zoom = min(21, max(0, zoom))

                tile_url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?"
                    f"center={center_lat},{center_lng}"
                    f"&zoom={zoom}"
                    f"&size={IMAGE_SIZE}x{IMAGE_SIZE}&maptype=satellite"
                    f"&key={GOOGLE_MAPS_API_KEY}"
                )
                response = requests.get(tile_url)
                if response.status_code != 200:
                    print(f"⚠️ Failed tile {i},{j}")
                    continue

                tile_img = Image.open(BytesIO(response.content))
                if tile_img.mode != 'RGB':
                    tile_img = tile_img.convert('RGB')

                stitched_image.paste(tile_img, (j * IMAGE_SIZE, i * IMAGE_SIZE))

        # Save stitched image
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        stitched_filename = f'stitched_{timestamp}.jpg'
        stitched_image.save(stitched_filename)
        print(f"✅ Saved stitched image: {stitched_filename}")

        # Run YOLO on stitched image
        results = model(stitched_image)

        all_detections = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if class_id not in AIRPLANE_CLASS_IDS or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Map pixel box to lat/lng
                lat1 = north - (y1 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lat2 = north - (y2 / (IMAGE_SIZE * NUM_TILES)) * lat_span
                lng1 = west + (x1 / (IMAGE_SIZE * NUM_TILES)) * lng_span
                lng2 = west + (x2 / (IMAGE_SIZE * NUM_TILES)) * lng_span

                all_detections.append({
                    'class_id': class_id,
                    'confidence': conf,
                    'lat1': max(lat1, lat2),
                    'lat2': min(lat1, lat2),
                    'lng1': min(lng1, lng2),
                    'lng2': max(lng1, lng2)
                })

        return jsonify({
            'message': 'Detection complete with stitched image',
            'detections': all_detections
        })

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
