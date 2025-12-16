import cv2
import mediapipe as mp
import numpy as np
import math

# Inisialisasi modul mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Fungsi menghitung sudut antara tiga titik (A, B, C)
def calculate_angle(a, b, c):
    a = np.array(a)  # Titik A (pinggul)
    b = np.array(b)  # Titik B (lutut)
    c = np.array(c)  # Titik C (pergelangan kaki)

    # Vektor AB dan CB
    ab = a - b
    cb = c - b

    # Dot product dan panjang vektor
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)

    # Hindari pembagian nol
    if norm_ab == 0 or norm_cb == 0:
        return 0.0

    # Hitung sudut dalam derajat
    angle = math.degrees(math.acos(dot_product / (norm_ab * norm_cb)))
    return angle

# Buka kamera
cap = cv2.VideoCapture(0)

# Gunakan Pose dari Mediapipe
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Kembali ke BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ambil koordinat landmark
        try:
            landmarks = results.pose_landmarks.landmark

            # Titik kanan tubuh (pinggul, lutut, pergelangan kaki)
            hip = [landmarks[24].x, landmarks[24].y]
            knee = [landmarks[26].x, landmarks[26].y]
            ankle = [landmarks[28].x, landmarks[28].y]

            # Hitung sudut lutut
            angle = calculate_angle(hip, knee, ankle)

            # Tampilkan sudut di layar
            cv2.putText(image, f'Angle: {int(angle)} deg',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        # Gambar kerangka tubuh
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

        cv2.imshow('Pose Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()