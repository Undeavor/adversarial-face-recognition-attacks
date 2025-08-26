import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os

# --------------------------
# Settings
# --------------------------
USE_CAMERA = True  # True for webcam, False for single image
SINGLE_IMAGE_PATH = 'adv_bob.jpg'  # used if USE_CAMERA = False

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
    img = Image.open(image_path).convert('RGB').resize((160,160))
    face_tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float()
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

# Example: register new face (uncomment to register)
register_face('bob.jpg', 'Bob')
known_faces['Bob'] = torch.load('Bob.pt').to(device)

# Cosine similarity
cos = torch.nn.CosineSimilarity(dim=1)
THRESHOLD = 0.7

# --------------------------
# Capture setup
# --------------------------
if USE_CAMERA:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Press 'q' to quit.")
else:
    # Single image
    frame = cv2.imread(SINGLE_IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not load image {SINGLE_IMAGE_PATH}")
        exit()

# --------------------------
# Main loop / single image
# --------------------------
while True:
    if USE_CAMERA:
        ret, frame = cap.read()
        if not ret:
            break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Crop face safely
            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_pil = Image.fromarray(face).resize((160,160))
            face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).unsqueeze(0).float()
            face_tensor = (face_tensor / 127.5 - 1).to(device)

            # Compute embedding
            embedding = resnet(face_tensor)

            # Compare to known faces
            name = "Unknown"
            for known_name, known_embedding in known_faces.items():
                similarity = cos(embedding, known_embedding.unsqueeze(0)).mean().item()
                if similarity > THRESHOLD:
                    name = known_name
                    break

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow('Face Detection & Recognition', frame)

    if USE_CAMERA:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

if USE_CAMERA:
    cap.release()
cv2.destroyAllWindows()