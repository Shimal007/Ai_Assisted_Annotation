from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging
from typing import List, Dict, Tuple
import torch
from collections import defaultdict
import time
import random
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global model variables
models = {}

# Class-specific confidence thresholds and colors
CLASS_CONFIDENCE_THRESHOLDS = {
    'car': 0.20, 'airplane': 0.15, 'tree': 0.20, 'vehicle': 0.20,
    # Add other class-specific thresholds here
}
CLASS_COLORS = {
    'person': '#FF3333', 'bicycle': '#00BFFF', 'car': '#1E90FF', 'motorcycle': '#32CD32',
    'airplane': '#FFD700', 'bus': '#FF69B4', 'train': '#FF4500', 'truck': '#4169E1',
    'boat': '#00FA9A', 'traffic light': '#FF8C00', 'building': '#9370DB', 'tree': '#228B22',
    'road': '#CD853F', 'parking lot': '#DDA0DD', 'fence': '#8A2BE2', 'bridge': '#87CEEB',
    'vehicle': '#FFB6C1'
}

def multi_scale_detection(model, image: np.ndarray, scales: List[float], conf_threshold: float) -> List[Dict]:
    """Perform detection at multiple scales."""
    all_detections = []
    original_height, original_width = image.shape[:2]
    for scale in scales:
        if int(original_width * scale) < 320 or int(original_height * scale) < 320: continue
        scaled_image = cv2.resize(image, (int(original_width * scale), int(original_height * scale)), interpolation=cv2.INTER_LANCZOS4)
        results = model(scaled_image, conf=conf_threshold, iou=0.5, verbose=False)
        for result in results:
            if result.obb is not None:
                boxes, confs, clss = result.obb.xyxyxyxy.cpu().numpy(), result.obb.conf.cpu().numpy(), result.obb.cls.cpu().numpy()
                for box, conf, cls in zip(boxes, confs, clss):
                    scaled_box = box / scale
                    center_x, center_y = np.mean(scaled_box[:, 0]), np.mean(scaled_box[:, 1])
                    width = np.linalg.norm(scaled_box[1] - scaled_box[0])
                    height = np.linalg.norm(scaled_box[2] - scaled_box[1])
                    angle = np.arctan2(scaled_box[1][1] - scaled_box[0][1], scaled_box[1][0] - scaled_box[0][0])
                    all_detections.append({'center_x': center_x, 'center_y': center_y, 'width': width, 'height': height, 'angle': angle, 'confidence': float(conf), 'class': int(cls)})
    return all_detections

def ensemble_detection(models_dict: Dict, image: np.ndarray, conf_threshold: float) -> List[Dict]:
    """Run ensemble detection with multiple models (Optimized Version)."""
    all_detections = []
    scales = [0.8, 1.0, 1.25]
    for model_name, model in models_dict.items():
        if model is None: continue
        try:
            scale_detections = multi_scale_detection(model, image, scales, conf_threshold)
            for det in scale_detections: det['model'] = model_name
            all_detections.extend(scale_detections)
        except Exception as e:
            logger.warning(f"Model {model_name} failed during ensemble: {e}")
    logger.info(f"Ensemble generated {len(all_detections)} candidates from {len(models_dict)} models.")
    return all_detections

def calculate_oriented_iou(det1: Dict, det2: Dict) -> float:
    try:
        rect1 = ((det1['center_x'], det1['center_y']), (det1['width'], det1['height']), det1['angle'] * 180 / np.pi)
        rect2 = ((det2['center_x'], det2['center_y']), (det2['width'], det2['height']), det2['angle'] * 180 / np.pi)
        intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
        if intersection_points is None: return 0.0
        intersection_area = cv2.contourArea(intersection_points)
        union_area = (det1['width'] * det1['height']) + (det2['width'] * det2['height']) - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0
    except:
        return 0.0

def advanced_nms(detections: List[Dict], iou_threshold: float = 0.5, angle_threshold: float = 15.0) -> List[Dict]:
    if not detections: return []
    class_groups = defaultdict(list)
    for det in detections: class_groups[det['class']].append(det)
    final_detections = []
    for _, class_detections in class_groups.items():
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        while class_detections:
            current = class_detections.pop(0)
            keep.append(current)
            remaining = [det for det in class_detections if calculate_oriented_iou(current, det) < iou_threshold]
            class_detections = remaining
        final_detections.extend(keep)
    return final_detections

def filter_invalid_detections(detections: List[Dict], image_shape: Tuple, conf_threshold: float) -> List[Dict]:
    h, w = image_shape[:2]
    valid_detections = []
    if not models: return []
    any_model = list(models.values())[0] # Get any model for class names

    for det in detections:
        if not (0 <= det['center_x'] < w and 0 <= det['center_y'] < h): continue
        min_size = min(w, h) * 0.005
        max_size = min(w, h) * 0.8
        if not (min_size < det['width'] < max_size and min_size < det['height'] < max_size): continue
        
        class_name = any_model.names.get(det['class'], 'unknown')
        min_conf = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, conf_threshold)
        if det['confidence'] >= min_conf:
            valid_detections.append(det)
    return valid_detections

def load_models():
    """Load multiple YOLO models for detection."""
    global models
    model_files = ['yolov8m-obb.pt', 'yolov8l-obb.pt', 'yolov8x-obb.pt']
    for model_file in model_files:
        try:
            logger.info(f"Loading {model_file}...")
            model = YOLO(model_file)
            if torch.cuda.is_available(): model.to('cuda')
            models[model_file] = model
            logger.info(f"{model_file} loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load {model_file}: {e}")

load_models()

@app.route("/model/status", methods=["GET"])
def model_status():
    return jsonify({
        "models_loaded": len(models) > 0,
        "available_models": list(models.keys()),
        "cuda_available": torch.cuda.is_available(),
    })

@app.route("/api/classes", methods=["GET"])
def get_classes():
    if not models: return jsonify({"error": "No models loaded"}), 500
    return jsonify({"classes": list(list(models.values())[0].names.values())})

@app.route("/api/detect", methods=["POST"])
def detect_objects():
    if 'image' not in request.files or not models:
        return jsonify({"error": "Image file or models not available"}), 400
    
    image_file = request.files['image']
    confidence = float(request.form.get('confidence', 0.3))
    model_choice = request.form.get('model_choice', 'ensemble')

    try:
        start_time = time.time()
        image_bytes = image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Processing image: {image_file.filename} with model: {model_choice}")

        all_detections = []
        if model_choice == 'ensemble':
            all_detections = ensemble_detection(models, opencv_image, confidence)
        else:
            single_model = models.get(model_choice)
            if not single_model:
                return jsonify({"error": f"Model {model_choice} not found"}), 404
            scales = [0.8, 1.0, 1.25]
            all_detections = multi_scale_detection(single_model, opencv_image, scales, confidence)

        valid_detections = filter_invalid_detections(all_detections, opencv_image.shape, confidence)
        final_detections = advanced_nms(valid_detections, iou_threshold=0.4)
        
        any_model = list(models.values())[0]
        processed_detections = []
        for det in final_detections:
            class_name = any_model.names.get(det['class'], 'unknown')
            
            # --- START FIX: Convert all NumPy numbers to standard Python floats ---
            processed_detections.append({
                "class": class_name,
                "confidence": float(det['confidence']),
                "bbox": [
                    float(det['center_x']),
                    float(det['center_y']),
                    float(det['width']),
                    float(det['height']),
                    float(det['angle'] * 180 / np.pi)
                ],
                "color": CLASS_COLORS.get(class_name, "#FFFFFF")
            })
            # --- END FIX ---
        
        processing_time = time.time() - start_time
        logger.info(f"Found {len(processed_detections)} objects in {processing_time:.2f}s")

        return jsonify({
            "detections": processed_detections,
            "image_info": {"filename": image_file.filename, "width": opencv_image.shape[1], "height": opencv_image.shape[0]},
            "model_info": {"model_used": model_choice, "processing_time": round(processing_time, 2)}
        })

    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)