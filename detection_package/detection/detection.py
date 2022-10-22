from detection.utils import DataGenerator, read_annotation_lines
from detection.models import Yolov4
from detection.config import yolo_config

class Detector(object):
    
    """
    for fish detection powered by yolov4

    Quick start: (set_iou_threshold, set_score_threshold) -> build_model -> load_weights -> detect/detect_image
    """
    
    def __init__(self, class_name_path, weights_path="", iou_threshold=0.413, score_threshold=0.3) -> None:
        self.weights_path = weights_path
        self.class_name_path = class_name_path
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def set_iou_threshold(self, iou_threshold):
        """Set IoU threshold before build_model"""
        self.iou_threshold = iou_threshold
    
    def set_score_threshold(self, score_threshold):
        """Set Score threshold before build_model"""
        self.score_threshold = score_threshold
    
    def load_weights(self, weights_path=""):
        if weights_path != "":
            self.weights_path = weights_path
        self.model.yolo_model.load_weights(self.weights_path)

    def build_model(self):
        self.model = Yolov4(weight_path=None, class_name_path=self.class_name_path, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold)

    def detect_image(self, img):
        """
        Detect image
        return output_img, detections
        output_img: Image
        detections: 
        {0: {'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class_name': class_name,
            'score': score,
            'w': x2-x1,
            'h': y2-y1}, 
         ...
        }
        """
        if self.model is None:
            print("Build Model First")
            return False
        output_img, detections_df = self.model.predict_img(img)
        detections = detections_df.T.to_dict()
        return output_img, detections

    def detect(self, img_path):
        """
        Detect image from image path.
        return output_img, detections  
        output_img: Image  
        detections: 
        {0: {'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class_name': class_name,
            'score': score,
            'w': x2-x1,
            'h': y2-y1}, 
         ...
        }
        """
        if self.model is None:
            print("Build Model First")
            return False
        output_img, detections_df = self.model.predict(img_path)
        detections = detections_df.T.to_dict()
        return output_img, detections

    def detect_video(self):
        """not yet, QQ"""
        pass