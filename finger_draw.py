import os
import time
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (one-time, ~10 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(options)

WRIST = 0
THUMB_TIP = 4
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_PAIRS = [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]

WINDOW_NAME = "Air Draw - press q to quit"

COLORS = [
    (255, 229, 0),   
    (146, 46, 255),   
    (63, 210, 255),   
    (107, 255, 124),  
    (255, 255, 255),  
]
current_color = COLORS[0]
FIST_MARGIN = 1.05          
PINCH_THRESHOLD = 0.055     
SWATCH_RADIUS = 14
SWATCH_GAP = 40
SWATCH_Y = 30

last_point = None
canvas = None                 
swatch_positions = []         

GESTURE_STABILITY_FRAMES = 6
stable_mode = "idle"
pending_mode = "idle"
pending_count = 0


def normalized_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_fist(landmarks):
    """A fist = all four (non-thumb) fingertips are closer to the wrist
    than their own knuckle is. Orientation-independent, unlike comparing
    raw y-coordinates."""
    wrist = landmarks[WRIST]
    curled = 0
    for tip_idx, mcp_idx in FINGER_PAIRS:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_mcp = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
        if d_tip < d_mcp * FIST_MARGIN:
            curled += 1
    return curled >= 3


def palm_center(landmarks, w, h):
    idxs = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return int((sum(xs) / len(xs)) * w), int((sum(ys) / len(ys)) * h)


def mouse_callback(event, x, y, flags, param):
    global current_color
    if event == cv2.EVENT_LBUTTONDOWN:
        for cx, cy, bgr in swatch_positions:
            if (x - cx) ** 2 + (y - cy) ** 2 <= (SWATCH_RADIUS + 6) ** 2:
                current_color = bgr
                break


def draw_hud(frame, status, brush_size):
    h, w = frame.shape[:2]

    status_colors = {
        "ERASING": (255, 255, 255),
        "DRAWING": (0, 255, 255),
        "PEN UP": (0, 200, 200),
        "TRACKING": (0, 200, 0),
        "NO HAND": (120, 120, 120),
    }
    color = status_colors.get(status, (120, 120, 120))

    cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Color swatches (also clickable, see mouse_callback)
    swatch_positions.clear()
    swatch_x = w - (len(COLORS) - 1) * SWATCH_GAP - 30
    for i, bgr in enumerate(COLORS):
        cx = swatch_x + i * SWATCH_GAP
        cv2.circle(frame, (cx, SWATCH_Y), SWATCH_RADIUS, bgr, -1)
        if bgr == current_color:
            cv2.circle(frame, (cx, SWATCH_Y), SWATCH_RADIUS + 4, (255, 255, 255), 2)
        swatch_positions.append((cx, SWATCH_Y, bgr))

    cv2.putText(frame, f"brush: {brush_size}px", (16, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "click a dot = color   pinch = draw   open = pen up   fist = erase   c clear   s save   q quit",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    global last_point, canvas, current_color, stable_mode, pending_mode, pending_count

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that it's connected and not in use.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    cv2.createTrackbar("Brush Size", WINDOW_NAME, 6, 40, lambda v: None)

    print("Air Draw running. Focus the window and press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, so movement feels natural
        h, w = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Trackbar can't go below 1px; clamp to a sane minimum
        brush_size = max(2, cv2.getTrackbarPos("Brush Size", WINDOW_NAME))
        eraser_size = brush_size * 4

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_present = False
        lm = None

        if result.hand_landmarks:
            hand_present = True
            lm = result.hand_landmarks[0]
            raw_mode = "erase" if is_fist(lm) else "draw"
        else:
            raw_mode = "idle"

        # Debounce: only commit to a new mode once it's held for several
        # consecutive frames in a row.
        if raw_mode == pending_mode:
            pending_count += 1
        else:
            pending_mode = raw_mode
            pending_count = 1

        if pending_count >= GESTURE_STABILITY_FRAMES and stable_mode != pending_mode:
            stable_mode = pending_mode
            last_point = None  # avoid a stray line jumping between modes

        mode = stable_mode

        status = "NO HAND"

        if hand_present and raw_mode == mode == "erase":
            status = "ERASING"
            x, y = palm_center(lm, w, h)
            if last_point is not None:
                cv2.line(canvas, last_point, (x, y), (0, 0, 0), eraser_size, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), eraser_size // 2, (0, 0, 0), -1)
            last_point = (x, y)
            cv2.circle(frame, (x, y), eraser_size // 2, (255, 255, 255), 2)
        elif hand_present and raw_mode == mode == "draw":
            index_tip = lm[INDEX_TIP]
            thumb_tip = lm[THUMB_TIP]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            touching = normalized_distance(index_tip, thumb_tip) < PINCH_THRESHOLD

            if touching:
                status = "DRAWING"
                if last_point is not None:
                    cv2.line(canvas, last_point, (x, y), current_color, brush_size, cv2.LINE_AA)
                last_point = (x, y)
                cv2.circle(frame, (x, y), 10, current_color, -1)
            else:
                # Pen lifted: move freely, don't connect the next strok
                status = "PEN UP"
                last_point = None
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
        elif hand_present:
            # No hand, or the live gesture doesn't match the locked mode
            status = "TRACKING"
            last_point = None
        else:
            last_point = None

        mask = canvas.astype(bool).any(axis=2)
        frame[mask] = canvas[mask]

        draw_hud(frame, status, brush_size)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            filename = f"drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()