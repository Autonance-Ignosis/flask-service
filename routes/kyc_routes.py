from flask import Blueprint, request, jsonify
from utils.aadhaar_extractor import extract_aadhaar_info_from_url
from utils.pan_extractor import extract_pan_info_from_url

kyc_bp = Blueprint('kyc', __name__)

@kyc_bp.route('/aadhaar', methods=['POST'])
def extract_aadhaar():
    data = request.get_json()
    image_url = data.get("url")
    if not image_url:
        return jsonify({'error': 'No URL provided'}), 400

    result = extract_aadhaar_info_from_url(image_url)
    return jsonify(result)


@kyc_bp.route('/pan', methods=['POST'])
def extract_pan():
    data = request.get_json()
    image_url = data.get("url")
    if not image_url:
        return jsonify({'error': 'No URL provided'}), 400

    result = extract_pan_info_from_url(image_url)
    return jsonify(result)


@kyc_bp.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'KYC service is running'})