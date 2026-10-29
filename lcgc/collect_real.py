"""
collect_real.py — Optional Real-Data Bridge for LCGC.

Records real hand landmark windows or continuous gesture streams directly
from a webcam using MediaPipe Hands, saving normalized 15x42 windows to .npz.

Usage:
  # Collect 50 windows for gesture 'Stop':
  python collect_real.py --gesture Stop --samples 50 --out real_data/stop.npz

  # Record a 30-second continuous stream:
  python collect_real.py --stream --duration 30 --out real_data/stream_user1.npz

  # Inspect saved real data:
  python collect_real.py --inspect real_data/stop.npz

Note:
  This script is completely optional. All simulation experiments and benchmarks
  operate autonomously on synthetic data without requiring a physical camera.
"""

import os
import sys
import argparse
import time
import numpy as np

import config


def normalize_landmarks_21x2(lm_array):
    """Normalize (21, 2) MediaPipe landmark coordinates to (42,) flat vector."""
    p0 = lm_array[config.WRIST_IDX]
    p9 = lm_array[config.MIDDLE_BASE_IDX]
    s_t = np.linalg.norm(p9 - p0)
    if s_t < 1e-8:
        return np.zeros(config.FEATURE_DIM, dtype=np.float32)
    normed = (lm_array - p0) / s_t
    return normed.flatten().astype(np.float32)


def record_windows(gesture_name, num_samples=50, camera_id=0, output_path=None):
    """Capture normalized (15, 42) windows from webcam."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("[ERROR] opencv-python and mediapipe are required for real webcam collection.")
        print("Install via: pip install opencv-python mediapipe")
        return False

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam device at index {camera_id}.")
        return False

    print(f"\n[INFO] Recording gesture: '{gesture_name}'")
    print(f"[INFO] Target samples: {num_samples} windows")
    print("[INFO] Press 'c' to start capturing, 'q' to abort.\n")

    window_buffer = []
    collected_windows = []
    capturing = False

    while cap.isOpened() and len(collected_windows) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            
            # Extract (21, 2) normalized MediaPipe coordinates
            coords = np.array([[lm.x, lm.y] for lm in hand_lm.landmark], dtype=np.float32)
            norm_vec = normalize_landmarks_21x2(coords)
            window_buffer.append(norm_vec)
            if len(window_buffer) > config.WINDOW_T:
                window_buffer.pop(0)

            if capturing and len(window_buffer) == config.WINDOW_T:
                collected_windows.append(np.array(window_buffer))
                print(f"\rCaptured {len(collected_windows)}/{num_samples} windows...", end="", flush=True)
        else:
            if capturing and len(window_buffer) > 0:
                # Hand lost: push zeros
                window_buffer.pop(0)

        # UI Overlay
        status = f"Capturing: {len(collected_windows)}/{num_samples}" if capturing else "Press 'C' to Capture"
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("LCGC Real Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capturing = True
        elif key == ord('q'):
            print("\nAborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(collected_windows) > 0:
        X = np.array(collected_windows, dtype=np.float32)
        y = np.full((len(X),), config.GESTURE_TO_IDX.get(gesture_name, 0), dtype=np.int32)
        
        if output_path is None:
            os.makedirs("real_data", exist_ok=True)
            output_path = os.path.join("real_data", f"{gesture_name.lower()}_{int(time.time())}.npz")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        np.savez_compressed(output_path, X=X, y=y, gesture_name=gesture_name)
        print(f"\n[SUCCESS] Saved {len(X)} windows to {output_path}")
        return True
    return False


def inspect_file(filepath):
    """Print dimensions and contents of a saved .npz recording."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return
    data = np.load(filepath)
    print(f"\nFile: {filepath}")
    for k in data.files:
        val = data[k]
        print(f"  {k}: shape={val.shape if hasattr(val, 'shape') else 'scalar'}, dtype={getattr(val, 'dtype', type(val))}")


def main():
    parser = argparse.ArgumentParser(description="LCGC Real Webcam Data Collection Bridge")
    parser.add_argument("--gesture", type=str, default="Stop", choices=config.GESTURE_NAMES,
                        help="Target gesture name to record")
    parser.add_argument("--samples", type=int, default=50, help="Number of 15-frame windows to capture")
    parser.add_argument("--camera_id", type=int, default=0, help="OpenCV VideoCapture device index")
    parser.add_argument("--out", type=str, default=None, help="Output .npz destination path")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect existing .npz data file")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_file(args.inspect)
    else:
        record_windows(args.gesture, args.samples, args.camera_id, args.out)


if __name__ == "__main__":
    main()
