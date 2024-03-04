import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNumber=0):
        lmList = []
        if results := self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
            myHand = results[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
        return lmList

def main():
    # Use camera index 0 (default camera)
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector
    detector = HandDetector()

    pTime = 0

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        while True:
            success, img = cap.read()
            img = detector.findHands(img)
            lmList = detector.findPosition(img)

            if lmList:
                print(lmList)  # Output the list of landmark positions

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            if not success:
                print("Error: Could not read frame.")
                break

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
