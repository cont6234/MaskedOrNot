import numpy as np
import cv2
import onnxruntime
import vision.utils.box_utils_numpy as box_utils


class FaceDetector(object):
    def __init__(self, model_filepath):
        self.model_path = model_filepath
        self.session = onnxruntime.InferenceSession(self.model_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = 0.0

    # image preprocessed accordingly before detection
    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (320, 240))
        image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    # pick which boxes to consider after detection
    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    # detection perfomed
    def detect_faces(self, frame):
        input_img = self.preprocess(frame)
        input_name = self.input_name
        confidences, boxes = self.session.run(None, {input_name: input_img})
        boxes, labels, probs = self.predict(frame.shape[1], frame.shape[0], confidences, boxes, self.threshold)
        boxes = boxes.tolist()
        # if no face detected, returns empty list
        if len(boxes) == 0:
            return []
        boxes_ = []
        for i in range(0, len(boxes)):
            box = boxes[i]
            box_ = {'start_x': int(box[0]), 'start_y': int(box[1]), 'end_x': int(box[2]), 'end_y': int(box[3])}
            boxes_.append(box_)
        return boxes_
