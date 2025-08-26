import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os

# --------------------------
# Device
# --------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Initialize MTCNN and FaceNet
# --------------------------
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------------------------
# Function to register a face from image
# --------------------------
def register_face(image_path, name):
    img = Image.open(image_path).convert('RGB').resize((160, 160))
    face_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    face_tensor = (face_tensor / 127.5 - 1).to(device)
    embedding = resnet(face_tensor)
    torch.save(embedding.detach().cpu(), f"{name}.pt")
    print(f"{name} registered with embedding saved to {name}.pt")
    return embedding.to(device)

# --------------------------
# Load known faces from .pt files
# --------------------------
known_faces = {}
for file in os.listdir('.'):
    if file.endswith('.pt'):
        name = file.replace('.pt','')
        known_faces[name] = torch.load(file).to(device)

# Example: register new face
register_face('bob.jpg', 'Bob')
known_faces['Bob'] = torch.load('Bob.pt').to(device)

# --------------------------
# Cosine similarity
# --------------------------
cos = torch.nn.CosineSimilarity(dim=1)
THRESHOLD = 0.7

# --------------------------
# Function to detect and recognize face
# --------------------------
def detect_and_recognize(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        boxes, _ = mtcnn.detect(rgb_frame)
    except Exception as e:
        print(f"Warning: MTCNN detection failed: {e}")
        return frame  # return original frame

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Crop face safely
            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_pil = Image.fromarray(face).resize((160, 160))
            face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float()
            face_tensor = (face_tensor / 127.5 - 1).to(device)

            # Compute embedding
            embedding = resnet(face_tensor)
            embedding = embedding.view(1, -1)  # Ensure shape (1,512)

            # Compare to known faces
            name = "Unknown"
            for known_name, known_embedding in known_faces.items():
                known_embedding = known_embedding.view(1, -1)
                similarity = cos(embedding, known_embedding).item()
                if similarity > THRESHOLD:
                    name = known_name
                    break

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# --------------------------
# Test by moving image over background
# --------------------------
background = cv2.imread('background.jpg')  # Load a background image
foreground = cv2.imread('Blankrune.png')  # Load the face image to slide
fg_h, fg_w, _ = foreground.shape
bg_h, bg_w, _ = background.shape

step_size = 50
for y in range(0, bg_h - fg_h, step_size):
    for x in range(0, bg_w - fg_w, step_size):
        test_frame = background.copy()
        test_frame[y:y+fg_h, x:x+fg_w] = foreground
        output_frame = detect_and_recognize(test_frame)
        cv2.imshow('Sliding Test', output_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()