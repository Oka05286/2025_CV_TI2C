import torch
import cv2
import os
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model
# =========================
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

# =========================
# Load image
# =========================
image_path = r"C:\Users\HP VICTUS GAMING 15\Documents\Visi Komputer\tugas_besar\IMG_20241211_150622.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
tensor = F.to_tensor(rgb).to(device)

# =========================
# Inference
# =========================
start = time.time()
with torch.no_grad():
    output = model([tensor])[0]
infer_time = time.time() - start

boxes = output["boxes"].cpu().numpy()
labels = output["labels"].cpu().numpy()
scores = output["scores"].cpu().numpy()

# =========================
# Detection
# =========================
threshold = 0.6
person_count = 0

for box, label, score in zip(boxes, labels, scores):
    if score < threshold:
        continue
    if label == 1:  # COCO person
        person_count += 1
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# =========================
# Footer bar (bawah)
# =========================
h, w, _ = image.shape
bar_height = 70

overlay = image.copy()
cv2.rectangle(
    overlay,
    (0, h - bar_height),
    (w, h),
    (0, 0, 0),
    -1
)

image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

footer_text = f"Model: Faster R-CNN | Orang: {person_count} | Time: {infer_time:.3f}s"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2

(text_w, text_h), _ = cv2.getTextSize(footer_text, font, font_scale, thickness)
text_x = (w - text_w) // 2
text_y = h - (bar_height // 2) + (text_h // 2)

cv2.putText(
    image,
    footer_text,
    (text_x, text_y),
    font,
    font_scale,
    (0, 255, 255),
    thickness,
    cv2.LINE_AA
)

# =========================
# Resize (opsional, konsisten dengan YOLO)
# =========================
max_height = 720
scale = max_height / image.shape[0]
image = cv2.resize(
    image,
    (int(image.shape[1] * scale), max_height)
)

# =========================
# Save output (nama mengikuti gambar asli)
# =========================
os.makedirs("outputs", exist_ok=True)

base_name = os.path.basename(image_path)
out_path = os.path.join("outputs", f"faster_rcnn_{base_name}")

cv2.imwrite(out_path, image)

print("Faster R-CNN output saved to:", out_path)