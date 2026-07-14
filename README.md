# LazerGraft Prototype (Air Draw) ✍️

Draw in thin air with nothing but webcam and hand.

![Air Draw in action](images/airdraw_demo.gif)

## Inspiration

This project is inspired by those online teachers who write on a transparent glass board while facing the camera . The writing seems to hang in the air in front of them, perfectly legible, like magic. **Air Draw** recreates that effect at home: instead of a glass board and a marker, your hand becomes the pen and the "ink" is drawn live on top of your webcam feed, letting you sketch, annotate, or teach as if you were writing directly in the air.

## How It Works

Air Draw uses [MediaPipe](https://developers.google.com/mediapipe)'s Hand Landmarker to track 21 points on your hand in real time. It watches for a few simple gestures and turns them into drawing actions:

| Gesture | Action |
|---|---|
| ✌️ Pinch (thumb + index finger touching) | **Draw** — moves the pen and leaves a trail |
| 🖐️ Open hand, fingers apart | **Pen up** — moves without drawing |
| ✊ Closed fist | **Erase** — wipes the canvas near your palm |
| 🖱️ Click a color dot on screen | **Change color** |


## Requirements

- Python 3.9+
- A webcam
- The following Python packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`

## Installation

1. Clone or download this project.
2. Install the dependencies:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Run the script:

   ```bash
   python air_draw.py
   ```

   On first run, the hand-tracking model (`hand_landmarker.task`, ~10 MB) will be downloaded automatically into the project folder.

> **Note:** The script currently opens webcam index `1` (`cv2.VideoCapture(1)`). If you only have one camera, or it doesn't open, change this to `0` in the code.
