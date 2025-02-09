import os
import io

from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# =============== 모델 로드 ===============
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# 모델 파일 경로 및 모델 로드
colors_classification_model = YOLO(os.path.join(base_dir, "colors_classification_11s_v1.pt"))
fruits_detection_model = YOLO(os.path.join(base_dir, "fruits_detection_11s_v1.pt"))
animals_detection_model = YOLO(os.path.join(base_dir, "animals_detection_11s_v1.pt"))


# =============== 유틸 함수 ===============
def read_image_as_cv2(file) -> np.ndarray:
    """
    Flask로 받은 'file'(werkzeug FileStorage)을 OpenCV 이미지(ndarray)로 변환
    """
    try:
        image_bytes = file.read()  # 바이트로 읽기
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # OpenCV는 BGR을 사용하므로, PIL(RGB) 이미지를 numpy array로 변환 후 COLOR_RGB2BGR 적용
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_cv2
    except Exception as e:
        raise ValueError(f"이미지 변환 실패: {e}")


def classify_center_region(model, image_cv2, resize: int = 224, threshold: float = 0.1) -> list:
    """
    이미지에서 9등분 영역(3x3) 중 '정중앙' 부분을 잘라 분류 모델로 예측  
    반환: [{"prediction_result": 결과, "confidence": 신뢰도}, ...]  
    결과가 없으면 빈 리스트 반환
    """
    h, w, _ = image_cv2.shape
    center_crop = image_cv2[h // 3: 2 * h // 3, w // 3: 2 * w // 3, :]

    # 입력 사이즈에 맞게 리사이즈
    if resize:
        center_crop = cv2.resize(center_crop, (resize, resize), interpolation=cv2.INTER_AREA)

    # conf 인자는 모델에 따라 의미가 다를 수 있음
    results = model.predict(source=center_crop, conf=threshold)
    predictions = []

    if results and hasattr(results[0], "probs") and results[0].probs is not None:
        probs_tensor = results[0].probs.data  # torch.Tensor
        # 상위 2개 예측 결과 추출
        topk_values, topk_indices = torch.topk(probs_tensor, k=2)
        top1_idx = int(topk_indices[0])
        top1_conf = float(topk_values[0])
        top2_idx = int(topk_indices[1])
        top2_conf = float(topk_values[1])

        if hasattr(results[0], "names") and results[0].names:
            top1_result = results[0].names[top1_idx]
            top2_result = results[0].names[top2_idx]
        else:
            top1_result = str(top1_idx)
            top2_result = str(top2_idx)

        # threshold 값보다 낮은 경우 필터링
        if top1_conf >= threshold:
            predictions.append({
                "prediction_result": top1_result,
                "confidence": round(top1_conf, 3)
            })
        if top2_conf >= threshold:
            predictions.append({
                "prediction_result": top2_result,
                "confidence": round(top2_conf, 3)
            })

    return predictions


def obj_detection(model, image_cv2, conf_thres: float = 0.25, resize_width: int = 640, resize_height: int = 640):
    """
    객체 탐지 수행 후,
      1. 바운딩 박스가 그려진 결과 이미지 (OpenCV ndarray)
      2. 바운딩 박스 정보 리스트 (dict)를 반환
    """
    resized_img = cv2.resize(image_cv2, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    results = model.predict(source=resized_img, conf=conf_thres)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
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

            # 바운딩 박스 그리기 (옵션)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_img, f"{prediction_result} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return resized_img, detections


# =============== 라우트 ===============
@app.route("/")
def home():
    return "Team SAURUS Flask Server is running..."


@app.route("/predict", methods=["POST"])
def predict():
    # 파일 유무 체크
    if "image" not in request.files:
        app.logger.error("No 'image' in request.files")
        return jsonify({"error": "No image file"}), 400

    img_file = request.files["image"]
    model_name = request.args.get("model", "NONE")

    app.logger.info(f"POST /predict called - File: {img_file.filename}, Selected Model: {model_name}")

    try:
        img_cv2 = read_image_as_cv2(img_file)
    except Exception as e:
        app.logger.error(f"이미지 읽기 실패: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # 모델 분기
        if model_name.lower() == "colors":
            predictions = classify_center_region(
                model=colors_classification_model,
                image_cv2=img_cv2,
                resize=224
            )
            if not predictions:
                app.logger.error("Classification failed or no result.")
                return jsonify({"error": "Classification failed"}), 500

            app.logger.info("Classification results:")
            for pred in predictions:
                app.logger.info(pred)

            return jsonify({
                "success": True,
                "type": "classification",
                "predictions": predictions
            }), 200

        elif model_name.lower() == "fruits":
            result_img, detections = obj_detection(
                model=fruits_detection_model,
                image_cv2=img_cv2,
            )
            for det in detections:
                app.logger.info(f"Detected -> {det}")

            return jsonify({
                "success": True,
                "type": "detection",
                "detections": detections
            }), 200

        elif model_name.lower() == "animals":
            result_img, detections = obj_detection(
                model=animals_detection_model,
                image_cv2=img_cv2,
                conf_thres=0.25,
                resize_width=640,
                resize_height=640,
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
