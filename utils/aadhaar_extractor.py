import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
from io import BytesIO

def extract_aadhaar_info_from_url(image_url):
    try:
        response = requests.get(image_url)
        image_file = BytesIO(response.content)

        img = Image.open(image_file).convert('L')
        img_cv = np.array(img)
        _, img_thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
        img_resized = cv2.resize(img_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(img_resized)
    except Exception as e:
        return {'error': str(e)}

    text = pytesseract.image_to_string(img)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Same Aadhaar parsing logic...
    name = None
    dob = None
    aadhaar_number = None

    dob_candidates = []
    for i, line in enumerate(lines):
        if re.search(r'issued|aadhaar no', line.lower()):
            continue
        dob_match = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', line)
        if dob_match:
            dob_candidates.append((i, dob_match.group()))

    for idx, dob_str in dob_candidates:
        dob = dob_str
        for candidate in reversed(lines[max(0, idx - 2):idx]):
            if re.match(r'^[A-Za-z\s\.]{5,}$', candidate) and not re.search(r'dob|date|govt|india|address', candidate.lower()):
                name = candidate.title()
                break
        if name:
            break

    for line in lines[idx + 1:] if dob else lines:
        match = re.search(r'\d{4}\s?\d{4}\s?\d{4}', line)
        if match:
            candidate = match.group().replace(" ", "")
            if re.match(r'^[2-9]\d{11}$', candidate):
                aadhaar_number = candidate
                break

    return {
        'name': name or 'Not found',
        'dob': dob or 'Not found',
        'aadhaar_number': aadhaar_number or 'Not found'
    }
