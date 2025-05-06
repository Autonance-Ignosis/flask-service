import requests
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np
from io import BytesIO

def extract_pan_info_from_url(image_url):
    try:
        response = requests.get(image_url)
        image_file = BytesIO(response.content)

        img_cv = cv2.cvtColor(np.array(Image.open(image_file)), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        img = Image.fromarray(thresh)

        img_cv = np.array(img.convert('L'))
        _, img_thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
        img_resized = cv2.resize(img_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(img_resized)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        print("OCR Output:", text)

        pan_keywords = [
            "income", "tax", "govt", "government",
            "permanent", "account", "number", "department", "india"
        ]

        if not any(keyword in text.lower() for keyword in pan_keywords):
            return {"error": "Image does not appear to be a PAN card"}

        pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text)
        if pan_match:
            pan_number = pan_match.group()
            print(f"Detected PAN: {pan_number}")

            return {
                "pan_number": pan_number,
                "message": "PAN number successfully extracted"
            }
        else:
            return {"error": "PAN number not found in the image"}

    except Exception as e:
        return {'error': f'Image processing failed: {str(e)}'}
