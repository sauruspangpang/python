import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# =============== 모델 로드 ===============
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# size: 224
colors_classification_model = YOLO(os.path.join(base_dir, "colors_classification_11n_v4.pt"))

# size: 640
fruits_detection_model = YOLO(os.path.join(base_dir, "fruits_detection_11n_v2.pt"))

# size: 640
animals_detection_model = YOLO(os.path.join(base_dir, "animals_detection_11n_v2.pt"))


# =============== 유틸 함수 ===============
def read_image_as_cv2(file) -> np.ndarray:
    """
    Flask로 받은 'file'(werkzeug FileStorage)을 OpenCV 이미지(ndarray)로 변환
    """
    image_bytes = file.read()  # 바이트로 읽기
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # OpenCV는 BGR을 쓰므로, PIL(RGB)을 np.array로 변환 후 COLOR_RGB2BGR
    img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_cv2


def classify_center_region(model, image_cv2, resize = 224):
    """
    이미지에서 9등분 영역(3x3) 중 '정중앙' 부분을 잘라 YOLOv11 분류 모델로 예측
    반환: (prediction_result, confidence) or (None, None)
    """
    h, w, _ = image_cv2.shape
    # 중앙 영역 크롭: h // 3 ~ 2 * h // 3, w // 3 ~ 2 * w // 3
    center_crop = image_cv2[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, :]

    # 분류 입력 사이즈에 맞춰 리사이즈
    if resize:
        center_crop = cv2.resize(center_crop, (resize, resize), interpolation = cv2.INTER_AREA)

    # 모델 추론
    results = model.predict(source = center_crop)

    if (
        len(results) > 0
        and hasattr(results[0], "probs")
        and results[0].probs is not None
    ):
        probs_tensor = results[0].probs.data  # torch.Tensor
        top1_idx = int(torch.argmax(probs_tensor))
        top1_conf = float(probs_tensor[top1_idx])

        if hasattr(results[0], "names") and results[0].names:
            prediction_result = results[0].names[top1_idx]
        else:
            prediction_result = str(top1_idx)

        return prediction_result, top1_conf
    return None, None


def obj_detection(model, image_cv2, conf_thres = 0.25, resize_width = 640, resize_height = 640):
    """
    객체 탐지 수행 후,
    (1) 바운딩 박스가 그려진 결과 이미지 (OpenCV ndarray)
    (2) 바운딩 박스 정보 list(dict)를 반환
    """
    resized_img = cv2.resize(
        image_cv2, (resize_width, resize_height), interpolation = cv2.INTER_AREA
    )

    results = model.predict(source = resized_img, conf = conf_thres)
    detected_result = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1

            if hasattr(model, "names") and cls_id in model.names:
                prediction_result = model.names[cls_id]
            else:
                prediction_result = str(cls_id)

            detected_result.append({
                "prediction_result": prediction_result,
                "class_id": cls_id,
                "confidence": float(conf),
                "bounding_box_area": [x1, y1, x2, y2],
            })

            # 바운딩 박스 그리기 (옵션)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                resized_img,
                f"{prediction_result} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return resized_img, detected_result


# =============== 라우트 ===============
@app.route("/")
def home():
    return "Team SAURUS Flask Server is running..."


@app.route("/predict", methods = ["POST"])
def predict():
    # 'image' 키로 파일 받음
    if "image" not in request.files:
        print("No 'image' in request.files")
        return jsonify({"error": "No image file"}), 400

    img_file = request.files["image"]
    model_name = request.form.get(key = "model", default = "NONE")  # merge 후 전달받는 모델 이름

    # 로그 출력
    print(f"===== POST /predict called =====")
    print(f"File name      : {img_file.filename}")
    print(f"Selected Model : {model_name}")

    # 이미지 -> OpenCV
    try:
        img_cv2 = read_image_as_cv2(img_file)
    except Exception as e:
        print(f"[Error] Failed to read image: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    # 모델 분기
    try:
        if model_name.lower() == "colors":
            # 분류 로직
            prediction_result, confidence = classify_center_region(
                model = colors_classification_model,
                image_cv2 = img_cv2,
                resize = 224,
            )
            if prediction_result is None:
                # 분류 실패
                print("Classification failed or no result.")
                return jsonify({"error": "Classification failed"}), 500

            # 터미널 로깅
            print(f"Classification result: {prediction_result} (conf = {confidence:.2f})")

            # JSON 응답
            return jsonify({
                "success": True,
                "type": "classification",
                "prediction_result": prediction_result,
                "confidence": confidence,
            }), 200

        elif model_name.lower() == "fruits":
            # 객체 감지 로직
            result_img, detection_list = obj_detection(
                model = fruits_detection_model,
                image_cv2 = img_cv2,
            )

            # 터미널 로깅
            for det in detection_list:
                print(f"Detected -> {det}")

            return jsonify({
                "success": True,
                "type": "detection",
                "detections": detection_list,
            }), 200

        elif model_name.lower() == "animals":
            # 객체 감지 로직
            result_img, detection_list = obj_detection(
                model = animals_detection_model,
                image_cv2 = img_cv2,
                conf_thres = 0.25,
                resize_width = 640,
                resize_height = 640,
            )

            # 터미널 로깅
            for det in detection_list:
                print(f"Detected -> {det}")

            return jsonify({
                "success": True,
                "type": "detection",
                "detections": detection_list,
            }), 200

        else:
            # 지원되지 않는 모델 이름
            print(f"Unsupported model: {model_name}")
            return jsonify({"error": f"Unsupported model '{model_name}'"}), 400

    except Exception as e:
        print(f"[Error] Model inference failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Flask 서버 실행
    app.run(host = "0.0.0.0", port = 6945, debug = True)
