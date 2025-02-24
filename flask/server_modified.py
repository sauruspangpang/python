import os
import io
from datetime import datetime
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# =============== 기본 디렉터리 ===============
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# =============== 모델 로드 ===============
colors_classification_model = YOLO(os.path.join(base_dir, "colors_classification_11s_v1.pt"))
fruits_detection_model = YOLO(os.path.join(base_dir, "fruits_detection_11s_v1.pt"))
animals_detection_model = YOLO(os.path.join(base_dir, "animals_detection_11s_v1.pt"))

# =============== 유틸 함수 ===============
def read_image_as_cv2(file) -> np.ndarray:
    """
    Flask에서 받은 FileStorage 객체를 OpenCV 이미지(ndarray)로 변환합니다.
    """
    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_cv2
    except Exception as e:
        raise ValueError(f"이미지 변환 실패: {e}")

def crop_and_resize(image_cv2, output_size=(640, 640)):
    """
    입력 이미지의 짧은 쪽 20~80% 영역을 crop한 후, 지정된 크기로 리사이즈합니다.
    """
    h, w, _ = image_cv2.shape
    crop_ratio_min = 0.2
    crop_ratio_max = 0.8

    if w < h:
        crop_x1 = int(w * crop_ratio_min)
        crop_x2 = int(w * crop_ratio_max)
        crop_width = crop_x2 - crop_x1
        center_y = h // 2
        half_crop = crop_width // 2
        crop_y1 = max(center_y - half_crop, 0)
        crop_y2 = min(center_y + half_crop, h)
    else:
        crop_y1 = int(h * crop_ratio_min)
        crop_y2 = int(h * crop_ratio_max)
        crop_height = crop_y2 - crop_y1
        center_x = w // 2
        half_crop = crop_height // 2
        crop_x1 = max(center_x - half_crop, 0)
        crop_x2 = min(center_x + half_crop, w)

    cropped = image_cv2[crop_y1:crop_y2, crop_x1:crop_x2]
    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return resized

def classify_center_region(model, image_cv2, resize=224, threshold=0.2) -> list:
    """
    이미지 중앙(전체 크기의 33~66% 영역)을 잘라 분류 모델로 예측합니다.
    결과는 [{"prediction_result": 결과, "confidence": 신뢰도}, ...] 형식으로 반환합니다.
    """
    h, w, _ = image_cv2.shape
    center_crop = image_cv2[h // 3: 2 * h // 3, w // 3: 2 * w // 3]

    if resize:
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
                result_name = results[0].names[idx] if hasattr(results[0], "names") and results[0].names else str(idx)
                predictions.append({
                    "prediction_result": result_name,
                    "confidence": round(conf_val, 3)
                })
    return predictions

def obj_detection(model, image_cv2, conf_thres=0.25, resize_width=640, resize_height=640):
    """
    객체 탐지를 수행합니다.
    반환:
        - 바운딩 박스가 그려진 결과 이미지 (OpenCV ndarray)
        - 탐지 정보 리스트 (dict 형식)
    """
    resized_img = cv2.resize(image_cv2, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    results = model.predict(source=resized_img, conf=conf_thres)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            if hasattr(model, "names") and model.names and cls_id in model.names:
                prediction_result = model.names[cls_id]
            else:
                prediction_result = str(cls_id)

            detections.append({
                "prediction_result": prediction_result,
                "class_id": cls_id,
                "confidence": round(float(conf), 3),
                "bounding_box_area": [x1, y1, x2, y2],
            })

            # 바운딩 박스 및 라벨 그리기
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_img, f"{prediction_result} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return resized_img, detections

def process_detection(model, image_cv2, conf_thres=0.2, resize=(640, 640)):
    """
    객체 탐지 모델을 사용하여 이미지를 처리합니다.
    반환: 결과 이미지, 탐지 정보 리스트
    """
    cropped_resized_img = crop_and_resize(image_cv2, output_size=resize)
    result_img, detections = obj_detection(model=model,
                                           image_cv2=cropped_resized_img,
                                           conf_thres=conf_thres,
                                           resize_width=resize[0],
                                           resize_height=resize[1])
    return result_img, detections

# =============== 라우트 ===============
@app.route("/")
def home():
    return "Team SAURUS Flask Server is running..."

@app.route("/predict", methods=["POST"])
def predict():
    # 이미지 파일 체크
    if "image" not in request.files:
        app.logger.error("No 'image' in request.files")
        return jsonify({"error": "No image file"}), 400

    img_file = request.files["image"]
    model_name = request.args.get("model", "NONE").lower()
    app.logger.info(f"POST /predict called - File: {img_file.filename}, Selected Model: {model_name}")

    try:
        img_cv2 = read_image_as_cv2(img_file)
    except Exception as e:
        app.logger.error(f"이미지 읽기 실패: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    try:
        if model_name == "colors":
            predictions = classify_center_region(model=colors_classification_model, image_cv2=img_cv2, resize=224)
            if not predictions:
                app.logger.error("Classification failed or no result.")
                return jsonify({"error": "Classification failed"}), 500

            for pred in predictions:
                app.logger.info(pred)

            return jsonify({
                "success": True,
                "type": "classification",
                "predictions": predictions
            }), 200

        elif model_name in ["fruits", "animals"]:
            detection_model = fruits_detection_model if model_name == "fruits" else animals_detection_model
            _, detections = process_detection(
                model=detection_model,
                image_cv2=img_cv2,
                conf_thres=0.2
            )
            for det in detections:
                app.logger.info(f"Detected -> {det}")

            return jsonify({
                "success": True,
                "type": "detection",
                "detections": detections
            }), 200

        else:
            app.logger.error(f"Unsupported model: {model_name}")
            return jsonify({"error": f"Unsupported model '{model_name}'"}), 400

    except Exception as e:
        app.logger.error(f"Model inference failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6945, debug=True)
