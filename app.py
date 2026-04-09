from flask import Flask, request, jsonify
import cv2
import numpy as np
import urllib.request

app = Flask(__name__)

# ฟังก์ชันดึงรูปจาก Firebase URL มาแปลงเป็นภาพ OpenCV
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.route('/analyze', methods=['POST'])
def analyze_wound():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # 1. โหลดรูปภาพ
        img = url_to_image(image_url)
        
        # 2. แปลงภาพเป็นโหมด HSV (เหมาะกับการแยกสีมากที่สุด)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 3. กำหนดช่วงสี (ค่าพวกนี้ปรับจูนได้ทีหลังให้แม่นขึ้น)
        # สีแดง (มี 2 ช่วงเพราะในวงล้อ HSV สีแดงอยู่ขอบพอดี)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # สีเหลือง (หนอง)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        # สีดำ (เนื้อตาย)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50]) # ค่าความสว่าง (V) ต่ำๆ คือสีดำ

        # 4. สร้างหน้ากาก (Mask) ดูดเฉพาะสีที่ต้องการ
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # 5. นับจำนวนพิกเซล (พื้นที่) ของแต่ละสี
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        black_pixels = cv2.countNonZero(mask_black)

        total_pixels = red_pixels + yellow_pixels + black_pixels

        # 6. คำนวณเปอร์เซ็นต์
        if total_pixels == 0:
            return jsonify({"red": 0, "yellow": 0, "black": 0})

        percent_red = round((red_pixels / total_pixels) * 100, 2)
        percent_yellow = round((yellow_pixels / total_pixels) * 100, 2)
        percent_black = round((black_pixels / total_pixels) * 100, 2)

        # 7. ส่งผลลัพธ์กลับไปให้ Unity
        return jsonify({
            "red": percent_red,
            "yellow": percent_yellow,
            "black": percent_black
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # เปิดเซิร์ฟเวอร์ที่พอร์ต 5000
    app.run(host='0.0.0.0', port=5000, debug=True)