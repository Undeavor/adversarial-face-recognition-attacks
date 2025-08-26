# Adversarial Face Generation & Recognition with FaceNet

This project demonstrates generating adversarial images targeting FaceNet embeddings, performing face recognition, and detecting faces in images or webcam streams using `facenet-pytorch` and `OpenCV`.

---

## Features

1. **FaceNet Embedding Generation**
   - Loads a target face and computes its embedding using `InceptionResnetV1`.
   - Saves embeddings for later use.

2. **Adversarial Image Generation**
   - Creates a random image and optimizes it to match the target face embedding.
   - Supports low-resolution simulation and MTCNN probability constraints.
   - Saves adversarial images that maximize cosine similarity to the target.

3. **Face Detection & Recognition**
   - Uses `MTCNN` to detect faces.
   - Computes embeddings and compares against known faces with cosine similarity.
   - Labels detected faces in images or live webcam feed.

4. **Sliding Test**
   - Slides a foreground image across a background to test recognition robustness.

5. **Webcam / Single Image Recognition**
   - Real-time face detection and recognition using webcam.
   - Can also process a single image for recognition.

---

## Installation

### Prerequisites
- Python 3.10+
- GPU recommended for faster computation (CUDA supported).

### Required Packages
```bash
pip install torch torchvision facenet-pytorch pillow matplotlib opencv-python numpy
```
