import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# =====================
# KONFIGURASI
# =====================
MODE = "squat"  # tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 80, 160  # ambang squat (deg)
DOWN_R, UP_R = 0.85, 1.00     # ambang push-up (rasio)
SAMPLE_OK = 4                 # minimal frame konsisten sebelum ganti state

# =====================
# INISIALISASI
# =====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

count, state = 0, "up"
debounce = deque(maxlen=6)

# =====================
# FUNGSI BANTU
# =====================
def find_angle(a, b, c):
    """Hitung sudut antar 3 titik (dalam derajat)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def ratio_pushup(lm):
    """Rasio jarak (shoulder–wrist)/(shoulder–hip) sisi kiri."""
    sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    wr = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
    hp = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

# =====================
# LOOP UTAMA
# =====================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    flag = None
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        if MODE == "squat":
            # Titik: Hip (23,24), Knee (25,26), Ankle (27,28)
            left_angle = find_angle(
                [lm[23].x, lm[23].y],
                [lm[25].x, lm[25].y],
                [lm[27].x, lm[27].y]
            )
            right_angle = find_angle(
                [lm[24].x, lm[24].y],
                [lm[26].x, lm[26].y],
                [lm[28].x, lm[28].y]
            )
            ang = (left_angle + right_angle) / 2.0
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"
            cv2.putText(frame, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:  # pushup
            r = ratio_pushup(lm)
            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"
            cv2.putText(frame, f"Ratio: {r:4.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Sistem debounce biar tidak menghitung ganda
        debounce.append(flag)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

        # Gambar landmark
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    cv2.putText(frame, f"Mode: {MODE.upper()} Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pose Counter (MediaPipe)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

cap.release()
cv2.destroyAllWindows()