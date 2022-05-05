#!/bin/python

from datetime import datetime

import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

from WebcamVideoStream import WebcamVideoStream


class Photomaton:

    def __init__(self):
        self.camera = None
        self.window_name = "window"
        self.previous_date = 0
        self.count = 5
        self.count_down_enable = False
        self.text = ""
        self.width = 1920
        self.height = 1080

    @staticmethod
    def put_text(frame, text, org=(230, 50)):
        cv2.putText(img=frame,
                    text=text,
                    org=org,
                    fontFace=cv2.QT_FONT_NORMAL,
                    fontScale=1.3,
                    color=(255, 88, 46),
                    thickness=3)

    def count_down(self, original_frame, frame):
        now = round(datetime.utcnow().timestamp() * 1000)
        if self.count > 0:
            if now > self.previous_date + 1000:
                self.count = self.count - 1
                self.previous_date = now
        else:
            cv2.rectangle(frame, (0, 0), (self.width, self.height), (255, 255, 255), -1)
            cv2.imwrite(f"image-{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", original_frame)
            self.count = 5
            self.count_down_enable = False

    def count_finger(self, frame, fingers):
        if fingers == [0, 1, 1, 0, 0]:
            if not self.count_down_enable:
                self.previous_date = round(datetime.utcnow().timestamp() * 1000)
            self.count_down_enable = True
        if fingers == [0, 1, 0, 0, 1]:
            self.put_text(frame, f"Stop")
            self.count_down_enable = False
            self.count = 5

    def show_webcam(self):
        self.camera = WebcamVideoStream(src=0, width=self.width, height=self.height).start()
        cv2.namedWindow(self.window_name, cv2.WND_PROP_VISIBLE)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        thumb_img = cv2.imread("images/instructions.png", cv2.IMREAD_UNCHANGED)
        face_detector = FaceDetector()
        hand_detector = HandDetector(detectionCon=0.5, maxHands=1)

        while True:
            original_frame = self.camera.read()
            if not self.camera.grabbed:
                print("ERROR: Could not grab frame")
                break
            if cv2.waitKey(30) & 0x7F == ord('q'):
                print("Exit requested.")
                return 0

            if self.count > 0:
                # Improve performance
                original_frame.flags.writeable = False
            frame = cv2.flip(original_frame, 1)
            frame.flags.writeable = False

            faces = face_detector.findFaces(img=frame, draw=False)
            if faces and len(faces) > 0 and len(faces[1]) > 0:
                for face in faces[1]:
                    bbox = face["bbox"]
                    cvzone.cornerRect(img=frame, bbox=bbox, l=10, t=2, colorR=(0, 150, 150), colorC=(255, 0, 255))

            hands = hand_detector.findHands(img=frame, draw=False, flipType=True)
            if hands:
                # Track on first Hand
                hand1 = hands[0]
                bbox1 = hand1["bbox"]
                cvzone.cornerRect(img=frame, bbox=bbox1, l=10, t=2, colorR=(0, 100, 200), colorC=(100, 200, 0))
                fingers = hand_detector.fingersUp(hand1)
                self.count_finger(frame, fingers)

            if self.count_down_enable:
                self.put_text(frame, f"{self.count}")
                self.count_down(original_frame, frame)

            frame = cvzone.overlayPNG(frame, thumb_img)
            frame.flags.writeable = False
            cv2.imshow(self.window_name, frame)
        self.camera.stop()
        cv2.destroyAllWindows()


def main():
    box = Photomaton()
    box.show_webcam()


if __name__ == '__main__':
    main()
