import cv2
import mediapipe as mp

mpFaces = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils


class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        # when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        # as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        # specified value then again it switches back to detection
        self.faces = mpFaces.FaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence)

    def find_faces(self, image, draw=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if draw:
            found_faces = self.faces.process(image)
            if found_faces.detections:
                for detection in found_faces.detections:
                    self.draw_face_detection(image, detection)

    @staticmethod
    def draw_box_label(image, text,
                       pos=(0, 0),
                       font_scale=0.6,
                       font_thickness=1,
                       text_color_bg=(0, 0, 0)):
        x, y = pos
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, pos, (x + text_w + 5, y - text_h - 10), text_color_bg, -1)
        cv2.putText(image, text, (x + 3, y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    def draw_face_detection(self, image, detection):
        boundingBox = detection.location_data.relative_bounding_box
        imgh, imgw, imgc = image.shape
        x, y, w, h = int(boundingBox.xmin * imgw), int(boundingBox.ymin * imgh), int(boundingBox.width * imgw), int(
            boundingBox.height * imgh)
        cv2.rectangle(image, (x, y), ((x + w), (y + h)), (150, 150, 150), 2)
        self.draw_box_label(image=image, text="visage", pos=(x - 1, y), text_color_bg=(150, 150, 150))
