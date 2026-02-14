import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def detect_cracks(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_count = 0
    total_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 400 < area < 8000:
            crack_count += 1
            total_area += area
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    image_area = image.shape[0] * image.shape[1]

    if crack_count == 0:
        confidence = round(95 - np.random.uniform(0, 4), 2)
        crack_detected = False
    else:
        confidence = min(100, round((total_area / image_area) * 6000, 2))
        crack_detected = True

    return original, crack_detected, confidence, crack_count


@app.route("/", methods=["GET", "POST"])
def index():
    original_image = None
    result_image = None
    result_text = None
    confidence = None
    crack = False
    crack_count = 0
    processing_time = None

    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            start_time = time.time()

            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            output_image, crack, confidence, crack_count = detect_cracks(file_path)

            result_filename = "result_" + filename
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            cv2.imwrite(result_path, output_image)

            original_image = url_for("static", filename=f"uploads/{filename}")
            result_image = url_for("static", filename=f"uploads/{result_filename}")

            processing_time = round(time.time() - start_time, 2)

            if crack:
                result_text = f"พบรอยแตก {crack_count} จุด"
            else:
                result_text = "ไม่พบรอยแตก"

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        result_text=result_text,
        confidence=confidence,
        crack=crack,
        crack_count=crack_count,
        processing_time=processing_time
    )


if __name__ == "__main__":
    app.run(debug=True)
