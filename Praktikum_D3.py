import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi FaceMesh dari Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Indeks mata kiri (berdasarkan model FaceMesh)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

def dist(p1, p2):
    """Hitung jarak Euclidean antara dua titik."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Variabel untuk kedipan
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3  # jumlah frame untuk dianggap berkedip
EYE_AR_THRESHOLD = 0.20  # ambang batas EAR untuk mata tertutup
is_closed = False

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Gunakan FaceMesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,  # aktifkan iris detection agar lebih akurat di sekitar mata
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Gambar hasil deteksi FaceMesh
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Ambil koordinat landmark mata kiri
                h, w, _ = frame.shape
                landmarks = face_landmarks.landmark
                L_top = (int(landmarks[L_TOP].x * w), int(landmarks[L_TOP].y * h))
                L_bottom = (int(landmarks[L_BOTTOM].x * w), int(landmarks[L_BOTTOM].y * h))
                L_left = (int(landmarks[L_LEFT].x * w), int(landmarks[L_LEFT].y * h))
                L_right = (int(landmarks[L_RIGHT].x * w), int(landmarks[L_RIGHT].y * h))

                # Hitung EAR (Eye Aspect Ratio)
                v = dist(L_top, L_bottom)
                h_ = dist(L_left, L_right)
                ear = v / (h_ + 1e-8)

                # Tampilkan nilai EAR di layar
                cv2.putText(frame, f"EAR(L): {ear:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # Logika hitung kedipan
                if ear < EYE_AR_THRESHOLD:
                    closed_frames += 1
                    if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                        blink_count += 1
                        is_closed = True
                else:
                    closed_frames = 0
                    is_closed = False

                # Tampilkan jumlah kedipan
                cv2.putText(frame, f"Blink: {blink_count}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Tampilkan video
        cv2.imshow("FaceMesh + EAR (tanpa cvzone)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()