import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Load image
img_path = 'tuntun.jpg'  # change to your image path
img = Image.open(img_path).convert('RGB')

# Detect faces
boxes, probs = mtcnn.detect(img)

# Check detection
if boxes is None:
    print("No face detected!")
else:
    for i, box in enumerate(boxes):
        print(f"Face {i+1}: Box={box}, Probability={probs[i]}")