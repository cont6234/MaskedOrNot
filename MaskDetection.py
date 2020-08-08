import tensorflow as tf
import numpy as np
import cv2
from FaceDetection import FaceDetector
import time


# fd_path = '/home/tigerit/PycharmProjects/MaskedOrNot/models/version-RFB-640.onnx'


class MaskDetector(object):
    def __init__(self, maskD_model_path, faceD_model_path):
        self.model_filepath = maskD_model_path
        self.face_detector = FaceDetector(faceD_model_path)
        self.interpreter, self.input_details, self.output_details = self.load_model_tflite(self.model_filepath)

    # loading the mask detection tflite model
    def load_model_tflite(self, model_path):
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    # preprocessing accordingly after face detection
    def preprocess(self, frame):
        t0 = time.time()
        boxes = self.face_detector.detect_faces(frame)
        print('UltraLightWeight() - face detection time: {:.3f} seconds'.format(time.time() - t0))
        # if no face detected, return none
        if len(boxes) == 0:
            return None
        # mask detection will be performed for the face with the max confidence score
        box = boxes[0]
        # box coordinates fixed accordingly if < 0. > height or > width
        if box['start_x'] < 0:
            box['start_x'] = 0
        if box['start_y'] < 0:
            box['start_y'] = 0
        if box['end_x'] > frame.shape[1]:
            box['end_x'] = frame.shape[1]
        if box['end_y'] > frame.shape[0]:
            box['end_y'] = frame.shape[0]

        face = frame[box['start_y']:box['end_y'], box['start_x']:box['end_x']]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = (np.float32(face) - 127.5) / 128
        face = np.expand_dims(face, axis=0)
        return face

    # return 'mask' and 'no-mask' percentages
    def masked_or_not(self, frame):
        input_data = self.preprocess(frame)
        # check for no face
        if input_data is None:
            print("no face detected...")
            return None
        t0 = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print('Mask detection time: {:.3f} seconds'.format(time.time() - t0))
        # return
        res = {'mask': float(output_data[0][0]), 'no-mask': float(output_data[0][1])}
        return res
