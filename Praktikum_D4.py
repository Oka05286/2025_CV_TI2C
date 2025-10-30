import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Fungsi bantu: menentukan jari yang terangkat
def fingers_up(hand_landmarks, handedness):
    """
    Mengembalikan list [jempol, telunjuk, tengah, manis, kelingking]
    1 = terbuka, 0 = tertutup
    """
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Jempol (horizontal axis)
    if handedness == "Right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # tangan kiri
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Empat jari lain (vertical axis)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


# Gunakan mediapipe hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    prev_count = -1
    stable_count = -1
    stability_frame = 0
    STABILITY_THRESHOLD = 3

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks,
                                                       results.multi_handedness):
                # Gambar kerangka tangan
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
                )

                # Jenis tangan (Right/Left)
                handedness_label = hand_handedness.classification[0].label

                # Hitung jari yang terangkat
                fingers = fingers_up(hand_landmarks, handedness_label)
                count = sum(fingers)

                # Stabilkan hasil deteksi
                if count == prev_count:
                    stability_frame += 1
                else:
                    stability_frame = 0

                if stability_frame >= STABILITY_THRESHOLD:
                    stable_count = count

                prev_count = count

                # Tampilkan teks
                cv2.putText(frame, f"{handedness_label} Hand", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {stable_count} {fingers}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Mediapipe Hands + Fingers", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()