from ultralytics import YOLO
import cv2
import os
import time

# =========================
# Model list (ringan -> berat)
# =========================
models = {
    "tiny_v8n": "yolov8n.pt",
    "small_v8s": "yolov8s.pt",
    "medium_v8m": "yolov8m.pt",
}

# =========================
# Load image
# =========================
img_path = r"C:\Users\HP VICTUS GAMING 15\Documents\Visi Komputer\tugas_besar\IMG_20241211_150622.jpg"
frame = cv2.imread(img_path)

if frame is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")

# =========================
# Output folder
# =========================
os.makedirs("outputs", exist_ok=True)

# =========================
# Inference loop
# =========================
for name, weight in models.items():
    print(f"\nRunning model: {name}")
    model = YOLO(weight)

    # --- inference time ---
    start = time.time()
    results = model(frame)
    infer_time = time.time() - start

    # --- count person ---
    boxes = results[0].boxes
    person_count = 0

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                person_count += 1

    # --- draw YOLO boxes ---
    annotated = results[0].plot()

    # =========================
    # Footer bar (bawah)
    # =========================
    h, w, _ = annotated.shape
    bar_height = 70

    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, h - bar_height),
        (w, h),
        (0, 0, 0),
        -1
    )

    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

    footer_text = f"Model: {name} | Orang: {person_count} | Time: {infer_time:.3f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (tw, th), _ = cv2.getTextSize(footer_text, font, font_scale, thickness)
    text_x = (w - tw) // 2
    text_y = h - (bar_height // 2) + (th // 2)

    cv2.putText(
        annotated,
        footer_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # =========================
    # Save result
    # =========================
    out_path = f"outputs/yolo_{name}.jpg"
    cv2.imwrite(out_path, annotated)

    print(f"Saved: {out_path}")