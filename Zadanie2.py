import urllib.request
import numpy as np
import cv2
from matplotlib import pyplot as plt


url = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg"
req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req = urllib.request.Request(url, headers=req_headers)

with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(gray.ravel(), 256, range=[0, 256], color="black")

plt.subplot(1, 2, 2)
for i, col in enumerate(["b", "g", "r"]):
  hist = cv2.calcHist([img], [i], None, [256], [0, 256])
  plt.plot(hist, color=col)
plt.show()


def assess_quality(image):
  g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  h = cv2.calcHist([g], [0], None, [256], [0, 256])
  total = g.size
  dark = (np.sum(h[0:21]) / total) * 100
  bright = (np.sum(h[235:256]) / total) * 100
  nz = np.where(h > 0)[0]
  rng = nz[-1] - nz[0] if len(nz) > 0 else 0

  status = "Prawidłowa"
  if dark > 40:
    status = "Wadliwe (Niedoświetlone)"
  elif bright > 40:
    status = "Wadliwe (Prześwietlone)"
  elif rng < 120:
    status = "Wadliwe (Niski kontrast)"

  print("Ocena:", status)


assess_quality(img)
