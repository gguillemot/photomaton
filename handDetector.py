import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils


class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        # when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        # as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        # specified value then again it switches back to detection
        self.hands = mpHands.Hands(max_num_hands=max_num_hands,
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def find_hand_landmarks(self, image, hand_number=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        found_hands = self.hands.process(image)
        landMarkList = []

        if found_hands.multi_handedness:
            label = found_hands.multi_handedness[hand_number].classification[0].label
            # Otherwise, thumb state is not correctly fetch
            label = "Right" if label == "Left" else "Left"

            if found_hands.multi_hand_landmarks:
                hand = found_hands.multi_hand_landmarks[hand_number]
                for hand_id, landMark in enumerate(hand.landmark):
                    imgH, imgW, imgC = originalImage.shape
                    xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                    landMarkList.append([id, xPos, yPos, label])
                if draw:
                    mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
        return landMarkList

    def count_fingers(self, image, hand_number=0, draw=False):
        hand_landmarks = self.find_hand_landmarks(image, hand_number, draw)
        count = 0
        if len(hand_landmarks) != 0:
            if hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] > hand_landmarks[3][1]:
                # Right Thumb
                count = count + 1
            elif hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] < hand_landmarks[3][1]:
                # Left Thumb
                count = count + 1
            if hand_landmarks[8][2] < hand_landmarks[6][2]:
                # Index finger
                count = count + 1
            if hand_landmarks[12][2] < hand_landmarks[10][2]:
                # Middle finger
                count = count + 1
            if hand_landmarks[16][2] < hand_landmarks[14][2]:
                # Ring finger
                count = count + 1
            if hand_landmarks[20][2] < hand_landmarks[18][2]:
                # Little finger
                count = count + 1
        return count
