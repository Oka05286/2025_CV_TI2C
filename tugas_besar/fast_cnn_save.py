import torch
import torchvision
from torchvision import transforms
import cv2
import os
import time
from PIL import Image
import numpy as np

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model pretrained MobileNetV2 (Fast CNN)
# =========================
model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
model.to(device)
model.eval()

# =========================
# Transform untuk MobileNet
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# =========================
# Load image
# =========================
image_path = r"C:\Users\HP VICTUS GAMING 15\Documents\Visi Komputer\tugas_besar\IMG_20241211_150622.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

h_img, w_img, _ = image.shape

# =========================
# Sliding window setup
# =========================
window_size = 128      # ukuran patch
stride = 64            # jarak langkah window
threshold = 0.5        # probabilitas minimal untuk dianggap "person"
person_count = 0

# Dummy ImageNet label untuk person
person_labels = [569,  # “jean” / clothing manusia (contoh) 
                 751]  # “suit” / clothing manusia (dummy)  
# Catatan: MobileNetV2 pretrained ImageNet tidak ada kelas 'person' murni. Ini hanya simulasi.

# =========================
# Sliding window inference
# =========================
start = time.time()
boxes = []

for y in range(0, h_img - window_size + 1, stride):
    for x in range(0, w_img - window_size + 1, stride):
        patch = image[y:y+window_size, x:x+window_size]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch_rgb)
        tensor = transform(patch_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probs, dim=0)
            if top_idx.item() in person_labels and top_prob.item() > threshold:
                boxes.append((x, y, x+window_size, y+window_size))
                person_count += 1

infer_time = time.time() - start

# =========================
# Draw bounding boxes
# =========================
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

# =========================
# Footer bar
# =========================
bar_height = 70
overlay = image.copy()
cv2.rectangle(overlay, (0, h_img - bar_height), (w_img, h_img), (0,0,0), -1)
image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
footer_text = f"Model: Fast CNN | Orang: {person_count} | Time: {infer_time:.3f}s"

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
(text_w, text_h), _ = cv2.getTextSize(footer_text, font, font_scale, thickness)
text_x = (w_img - text_w) // 2
text_y = h_img - (bar_height // 2) + (text_h // 2)
cv2.putText(image, footer_text, (text_x, text_y), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)

# =========================
# Resize output
# =========================
max_height = 720
scale = max_height / image.shape[0]
image = cv2.resize(image, (int(image.shape[1] * scale), max_height))

# =========================
# Save output
# =========================
os.makedirs("outputs", exist_ok=True)
base_name = os.path.basename(image_path)
out_path = os.path.join("outputs", f"fast_cnn_{base_name}")
cv2.imwrite(out_path, image)

print("Fast CNN output saved to:", out_path)
print(f"Orang terdeteksi: {person_count}, Waktu inferensi: {infer_time:.3f}s")