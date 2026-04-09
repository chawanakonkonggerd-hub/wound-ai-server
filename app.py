import cv2
import numpy as np
import urllib.request
from flask import Flask, request, jsonify

app = Flask(__name__)

def analyze_wound_hybrid(image_url):
    try:
        # 1. โหลดรูปภาพจาก Firebase URL
        req = urllib.request.urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        # 2. ค้นหา ArUco Marker (ตระกูล 4x4 รหัส 50 ที่พี่จะใช้)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        # ถ้าหากระดาษไม่เจอ ให้ส่ง Error กลับไปเตือนคนไข้
        if ids is None or len(corners) == 0:
            return {"error": "ไม่พบกระดาษอ้างอิง ArUco (5x5 ซม.) ในกรอบภาพ กรุณาถ่ายใหม่"}

        # 3. คำนวณสเกล (Pixel Calibration)
        marker_corners = corners[0][0] # ดึงพิกัดมุมทั้ง 4 ของกระดาษ
        marker_pixel_area = cv2.contourArea(marker_corners) # หาพื้นที่กระดาษเป็น "พิกเซล"
        
        # เรารู้ว่ากระดาษของจริงคือ 5x5 ซม. = 25 ตารางเซนติเมตร
        # ดังนั้น 1 ตารางเซนติเมตร = กี่พิกเซล?
        pixels_per_cm2 = marker_pixel_area / 25.0 

        # 🚨 4. วิธีแก้ปัญหาระดับเซียน: ถมสีขาวทับกระดาษดำทิ้งไปเลย! (Masking Out) 🚨
        # เราใช้คำสั่ง fillPoly วาดรูปหลายเหลี่ยมสีขาว (255,255,255) ทับลงไปตรงพิกัดกระดาษ
        cv2.fillPoly(img, [np.int32(marker_corners)], (255, 255, 255))

        # 5. กระบวนการดูดสีเนื้อเยื่อ (HSV Color Segmentation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # -- ดูดสีแดง (เนื้อดี / Granulation) --
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))

        # -- ดูดสีเหลือง (หนอง / Slough) --
        mask_yellow = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))

        # -- ดูดสีดำ (เนื้อตาย / Necrosis) --
        # (ตรงนี้ปลอดภัยแล้ว! เพราะกระดาษดำโดนถมขาวไปแล้ว AI จะจับแค่เนื้อตายบนแผลจริงๆ)
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))

        # 6. นับจำนวนพิกเซลของบาดแผลทั้งหมด
        red_px = cv2.countNonZero(mask_red)
        yellow_px = cv2.countNonZero(mask_yellow)
        black_px = cv2.countNonZero(mask_black)
        total_wound_px = red_px + yellow_px + black_px

        if total_wound_px == 0:
            return {"error": "ไม่พบเนื้อเยื่อบาดแผลสี แดง/เหลือง/ดำ ในภาพ"}

        # 7. คำนวณเปอร์เซ็นต์สี
        red_pct = (red_px / total_wound_px) * 100
        yellow_pct = (yellow_px / total_wound_px) * 100
        black_pct = (black_px / total_wound_px) * 100

        # 🌟 8. คำนวณขนาดแผลจริง! 🌟
        # เอาพิกเซลของแผลทั้งหมด หารด้วย อัตราส่วนไม้บรรทัดดิจิทัลที่เราทำไว้
        actual_wound_area_sqcm = total_wound_px / pixels_per_cm2

        # ส่งผลลัพธ์ทั้งหมดกลับไปให้แอป Unity
        return {
            "area_sqcm": round(actual_wound_area_sqcm, 2),
            "red": round(red_pct, 2),
            "yellow": round(yellow_pct, 2),
            "black": round(black_pct, 2),
            "error": ""
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400
    
    result = analyze_wound_hybrid(image_url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)