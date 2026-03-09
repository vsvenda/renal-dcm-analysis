import pydicom
import numpy as np
import imageio
from pathlib import Path

folder = Path("DICOM/00007F1D/AA0D871C/AA9E2850/00002C0A")

# Učitaj sve fajlove
slices = []
for f in folder.glob("*"):
    ds = pydicom.dcmread(f)
    slices.append(ds)

# Sortiraj po Z koordinati
slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

frames = []

for ds in slices:
    img = ds.pixel_array.astype(np.float32)

    # Konvertuj u HU
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    img = img * slope + intercept

    # Window za abdomen (venous)
    window_center = 40
    window_width = 400
    img = np.clip(img,
                  window_center - window_width//2,
                  window_center + window_width//2)

    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    frames.append(img)

imageio.mimsave("ct_preview.gif", frames, duration=0.03)

print("GIF saved as ct_preview.gif")