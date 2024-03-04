This module uses the Mediapipe library to perform hand tracking and extract hand landmarks using a webcam.

## Requirements
- Python 3.x
- OpenCV
- Mediapipe

## Installation
1. Install the required libraries:
   ```bash
   pip install opencv-python mediapipe

   Usage
The module initializes the camera (default index 0) and continuously tracks hands.
It displays the live camera feed with hand landmarks and FPS counter.


HandDetector Class
The HandDetector class provides a simple interface for hand tracking:

findHands(img, draw=True): Detects and draws hand landmarks on the image.
findPosition(img, handNumber=0): Returns a list of landmark positions for a specified hand.
