import logging
import warnings
import cv2
import os
import numpy
from flask import Flask, render_template, request, jsonify, Response, send_file
from MaskDetection import MaskDetector
from FaceDetection import FaceDetector

warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path="/static")

logger = logging.getLogger(__name__)

args = {'faceDetection_model': '/home/tigerit/PycharmProjects/MaskedOrNot/models/version-RFB-640.onnx',
        'maskDetection_model': '/home/tigerit/PycharmProjects/MaskedOrNot/models/mask_detector.tflite'}

faceDetector = FaceDetector(args['faceDetection_model'])
maskDetector = MaskDetector(args['maskDetection_model'], args['faceDetection_model'])


@app.route('/', methods=['GET'])
def home():
    return render_template('main.html')


@app.route('/detect_faces', methods=['POST'])
def detect():
    photo = request.get_data()
    # by specifying rank, desired number of detected faces can be returned, returns all if rank not provided
    rank = request.args.get('rank', default='-1').lower()
    try:
        data = numpy.frombuffer(photo, dtype=numpy.uint8)
        frame = cv2.imdecode(data, 1)
    except:
        return jsonify({'msg': 'Not a valid image!'})

    detected_faces = faceDetector.detect_faces(frame)
    if len(detected_faces) == 0:
        return jsonify({'msg': 'No face detected...'})
    elif rank:
        return jsonify(detected_faces[0:int(rank)])
    else:
        return jsonify(detected_faces)


@app.route('/detect_mask', methods=['POST'])
def mask_detection():
    photo = request.get_data()
    try:
        data = numpy.frombuffer(photo, dtype=numpy.uint8)
        frame = cv2.imdecode(data, 1)
    except:
        return jsonify({'msg': 'Not a valid image!'})

    out = maskDetector.masked_or_not(frame)
    if out is None:
        return jsonify({'msg': 'No face detected, mask detection not possible...'})
    else:
        return jsonify(out)


def main():
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), use_reloader=False, debug=True, threaded=True)


if __name__ == "__main__":
    main()
