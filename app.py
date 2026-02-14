from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def detect_crack(image_path):
    img = cv2.imread(image_path)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ปรับค่าได้ถ้าอยากให้เส้นชัดขึ้น
    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    crack_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # กรองเฉพาะเส้นขนาดเล็กถึงกลาง (ลักษณะรอยแตก)
        if 80 < area < 5000:
            crack_detected = True
            cv2.drawContours(original, [cnt], -1, (0, 0, 255), 2)

    return crack_detected, original


@app.route("/", methods=["GET", "POST"])
def index():

    original_image = None
    result_image = None
    result_text = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            unique_name = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(filepath)

            crack, processed_img = detect_crack(filepath)

            result_path = os.path.join(
                UPLOAD_FOLDER,
                "result_" + unique_name
            )
            cv2.imwrite(result_path, processed_img)

            original_image = "uploads/" + unique_name
            result_image = "uploads/" + "result_" + unique_name

            if crack:
                result_text = "พบรอยแตก"
            else:
                result_text = "ไม่พบรอยแตก"

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        result_text=result_text
    )


if __name__ == "__main__":
    app.run()

