import os
import io
from datetime import datetime  # 추가: 파일명에 날짜/시간을 붙이기 위해 필요
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# === 환경 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PORT = int(os.environ.get("FLASK_PORT", 6945))
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")

# 디버깅 용도로 결과 이미지를 저장할 디렉터리 생성
DETECTION_RESULTS_DIR = os.path.join(BASE_DIR, "detection_results")
os.makedirs(DETECTION_RESULTS_DIR, exist_ok=True)

# === 모델 로드 함수 및 중앙 관리 ===
def load_model(model_filename: str) -> YOLO:
    model_path = os.path.join(BASE_DIR, model_filename)
    return YOLO(model_path)

MODELS = {
    "colors": load_model("11n_cls_colors_v10.pt"),
    "fruits": load_model("11n_detection_fruits_v10.pt"),
    "animals": load_model("11n_detection_animals_v10.pt"),
    "transportation": load_model("11n_detection_transportation_v1.pt"),
    "stationery": load_model("11n_detection_stainonary_v2.pt"),
    "clothes": load_model("11n_detection_clothes_v1.pt")
}

# === 유틸리티 함수들 ===
def read_image_as_cv2(file) -> np.ndarray:
    """
    Flask FileStorage 객체를 OpenCV 이미지(ndarray)로 변환합니다.
    """
    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_cv2
    except Exception as e:
        raise ValueError(f"Image conversion failed: {e}")

def crop_and_resize(image_cv2: np.ndarray, output_size=(640, 640)) -> np.ndarray:
    """
    전체 이미지에서 짧은 변의 길이에 맞춰 중앙에 1:1 정사각형 영역을 크롭한 후,
    output_size 크기로 리사이즈합니다.
    """
    h, w, _ = image_cv2.shape
    side = min(h, w)
    center_y, center_x = h // 2, w // 2
    half_side = side // 2
    start_y = max(center_y - half_side, 0)
    start_x = max(center_x - half_side, 0)
    cropped = image_cv2[start_y:start_y + side, start_x:start_x + side]
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)

def classify_center_region(model: YOLO, image_cv2: np.ndarray, resize: int = 224, threshold: float = 0.2) -> list:
    """
    이미지 중앙(33~66% 영역)을 잘라 분류 모델로 예측합니다.
    반환 형식: [{"prediction_result": 결과, "confidence": 신뢰도}, ...]
    """
    h, w, _ = image_cv2.shape
    center_crop = image_cv2[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    center_crop = cv2.resize(center_crop, (resize, resize), interpolation=cv2.INTER_AREA)
    results = model.predict(source=center_crop, conf=threshold)
    predictions = []
    if results and hasattr(results[0], "probs") and results[0].probs is not None:
        probs_tensor = results[0].probs.data
        k = min(5, probs_tensor.size(0))
        topk_values, topk_indices = torch.topk(probs_tensor, k=k)
        for i in range(k):
            conf_val = float(topk_values[i])
            if conf_val >= threshold:
                idx = int(topk_indices[i])
                name = results[0].names[idx] if hasattr(results[0], "names") and results[0].names else str(idx)
                predictions.append({
                    "prediction_result": name,
                    "confidence": round(conf_val, 3)
                })
    return predictions

def obj_detection(model: YOLO, image_cv2: np.ndarray, conf_thres: float = 0.25,
                  resize_dims: tuple = (640, 640)) -> tuple[np.ndarray, list]:
    """
    객체 탐지를 수행합니다.
    반환:
        - 바운딩 박스가 그려진 이미지 (ndarray)
        - 탐지 정보 리스트 (dict 형식)
    """
    resized_img = cv2.resize(image_cv2, resize_dims, interpolation=cv2.INTER_AREA)
    results = model.predict(source=resized_img, conf=conf_thres)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            prediction_result = (
                model.names[cls_id] if hasattr(model, "names") and model.names and cls_id in model.names
                else str(cls_id)
            )
            detections.append({
                "prediction_result": prediction_result,
                "class_id": cls_id,
                "confidence": round(conf, 3),
                "bounding_box_area": [x1, y1, x2, y2],
            })
            # 바운딩 박스 및 라벨 그리기
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                resized_img,
                f"{prediction_result} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return resized_img, detections

def process_detection(model: YOLO, image_cv2: np.ndarray, conf_thres: float = 0.2,
                      resize_dims: tuple = (640, 640)) -> tuple[np.ndarray, list]:
    """
    객체 탐지 모델을 적용하기 위해 이미지를 crop 및 resize한 후, 탐지 수행
    """
    processed_img = crop_and_resize(image_cv2, output_size=resize_dims)
    return obj_detection(model=model, image_cv2=processed_img, conf_thres=conf_thres, resize_dims=resize_dims)

# === 새롭게 추가: 결과 이미지를 서버 내에 저장 (디버깅 용도) ===
def save_result_image(result_img: np.ndarray, prefix: str) -> None:
    """
    결과 이미지를 (prefix + 날짜/시간) 형태의 파일명으로 서버 내부에 저장.
    실제 응답(JSON)에는 포함하지 않고, 서버 디버깅 용도로만 사용.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{prefix}.jpg"
    filepath = os.path.join(DETECTION_RESULTS_DIR, filename)
    cv2.imwrite(filepath, result_img)
    app.logger.info(f"[DEBUG] Saved image to: {filepath}")

# === Flask 라우트 ===
@app.route("/")
def home():
    return "Team SAURUS Flask Server is running..."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        app.logger.error("No 'image' found in request.files")
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files["image"]
    model_name = request.args.get("model", "none").lower()
    app.logger.info(f"Received image: {img_file.filename}, Model: {model_name}")

    try:
        img_cv2 = read_image_as_cv2(img_file)
    except Exception as e:
        app.logger.error(f"Error reading image: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    try:
        if model_name == "colors":
            # 분류 모델 처리
            predictions = classify_center_region(model=MODELS["colors"], image_cv2=img_cv2, resize=224)
            if not predictions:
                app.logger.error("Classification returned no results.")
                return jsonify({"error": "Classification failed"}), 500

            # 로깅
            for pred in predictions:
                app.logger.info(pred)

            return jsonify({
                "success": True,
                "type": "classification",
                "predictions": predictions
            }), 200

        elif model_name in ["fruits", "animals", "transportation", "stationery", "clothes"]:
            # 객체 탐지 모델 처리
            detection_model = MODELS[model_name]
            result_img, detections = process_detection(
                model=detection_model,
                image_cv2=img_cv2,
                conf_thres=0.2
            )

            # 서버 내부에 결과 이미지를 저장 (JSON 응답에는 포함 X)
            save_result_image(result_img, prefix=model_name)

            # 탐지 결과 로그
            for det in detections:
                app.logger.info(f"Detected: {det}")

            return jsonify({
                "success": True,
                "type": "detection",
                "detections": detections
            }), 200

        else:
            app.logger.error(f"Unsupported model: {model_name}")
            return jsonify({"error": f"Unsupported model '{model_name}'"}), 400

    except Exception as e:
        app.logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
