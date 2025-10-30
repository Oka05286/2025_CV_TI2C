import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi modul Hands dari MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Fungsi menghitung jarak Euclidean antar titik
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Fungsi klasifikasi gesture
def classify_gesture(lm):
    wrist = np.array(lm[0][:2])
    thumb_tip = np.array(lm[4][:2])
    index_tip = np.array(lm[8][:2])
    middle_tip = np.array(lm[12][:2])
    ring_tip = np.array(lm[16][:2])
    pinky_tip = np.array(lm[20][:2])

    # Hitung rata-rata jarak dari ujung jari ke pergelangan
    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist),
        dist(thumb_tip, wrist)
    ])

    # Aturan sederhana berbasis jarak dan posisi
    if dist(thumb_tip, index_tip) < 35:
        return "OK"
    if (thumb_tip[1] < wrist[1] - 40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS_UP"
    if r_mean < 120:
        return "ROCK"
    if r_mean > 200:
        return "PAPER"
    if (
        dist(index_tip, wrist) > 180 and
        dist(middle_tip, wrist) > 180 and
        dist(ring_tip, wrist) < 160 and
        dist(pinky_tip, wrist) < 160
    ):
        return "SCISSORS"
    return "UNKNOWN"

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Aktifkan deteksi tangan dari MediaPipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Konversi warna ke RGB untuk MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                lm = []
                for id, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    lm.append((cx, cy))

                label = classify_gesture(lm)

                cv2.putText(frame, f"Gesture: {label}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # Gambar titik dan garis tangan
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        cv2.imshow("Hand Gestures (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()