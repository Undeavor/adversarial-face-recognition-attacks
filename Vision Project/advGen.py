import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ----------------------
# Device + models
# ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# ----------------------
# Load target embedding
# ----------------------
target_embedding = torch.load('Bob.pt').to(device)

# ----------------------
# Load base image
# ----------------------
base_img_path = 'base_face.jpg'
pil_base = Image.open(base_img_path).convert('RGB').resize((160,160))

img = torch.tensor(np.array(pil_base)).permute(2,0,1).unsqueeze(0).float().to(device)
img = (img / 127.5 - 1)   # normalize to [-1,1]
img.requires_grad = True

# ----------------------
# Detect face & create mask
# ----------------------
face_boxes, _ = mtcnn.detect(pil_base)
if face_boxes is not None:
    x1, y1, x2, y2 = [int(v) for v in face_boxes[0]]
    face_mask = torch.zeros_like(img)
    face_mask[:,:, y1:y2, x1:x2] = 1.0
    print(f"Mask applied to face region: {x1,y1,x2,y2}")
else:
    face_mask = torch.ones_like(img)
    print("Warning: No face detected, using full image mask.")

# ----------------------
# Optimizer
# ----------------------
optimizer = torch.optim.Adam([img], lr=0.03)  # higher LR for stronger updates
cos = torch.nn.CosineSimilarity(dim=1)
EPSILON = 1e-6

# ----------------------
# Optimization loop
# ----------------------
max_steps = 1000
for step in range(max_steps):
    optimizer.zero_grad()
    img_clamped = torch.clamp(img, -1, 1)

    # FaceNet embedding
    embedding = resnet(img_clamped)
    loss_face = 1 - cos(embedding, target_embedding).mean()

    # Low-res simulation
    pil_img = Image.fromarray(((img_clamped.squeeze(0).permute(1,2,0).detach().cpu().numpy()+1)*127.5).astype(np.uint8))
    pil_lowres = pil_img.resize((64,64), Image.BILINEAR).resize((160,160), Image.BILINEAR)
    img_lowres = torch.tensor(np.array(pil_lowres)).permute(2,0,1).unsqueeze(0).float().to(device)
    img_lowres = (img_lowres / 127.5 - 1)
    embedding_lowres = resnet(img_lowres)
    loss_lowres = 1 - cos(embedding_lowres, target_embedding).mean()

    # MTCNN detection
    try:
        _, probs = mtcnn.detect(pil_img)
        mtcnn_prob = float(probs.max()) if probs is not None else EPSILON
    except Exception as e:
        mtcnn_prob = EPSILON
        print(f"Warning: MTCNN detection failed at step {step}: {e}")

    # --- Loss balancing ---
    mtcnn_loss = -torch.log(torch.tensor(mtcnn_prob + EPSILON))
    weight_mtcnn = min(5.0, 0.5 + step/150)   # more aggressive growth
    total_loss = loss_face + 0.8*loss_lowres + weight_mtcnn*mtcnn_loss

    # Backprop
    total_loss.backward()

    # Restrict gradient to face only
    img.grad.data *= face_mask

    # Amplify gradients for visibility
    img.grad.data *= 4.0

    # Smooth gradients
    img.grad.data = F.avg_pool2d(img.grad.data, kernel_size=3, stride=1, padding=1)

    optimizer.step()
    img.data = torch.clamp(img.data, -1, 1)

    # Logging
    if step % 25 == 0:
        face_sim = 1 - loss_face.item()
        face_sim_low = 1 - loss_lowres.item()
        print(f"Step {step} - Sim(full): {face_sim:.4f}, Sim(lowres): {face_sim_low:.4f} - MTCNN prob: {mtcnn_prob:.4f}")

    # Early stop if both FaceNet and MTCNN are high
    if face_sim > 0.92 and face_sim_low > 0.85 and mtcnn_prob > 0.99:
        print(f"✅ Early stop at step {step}: FaceNet >0.92 and MTCNN >0.99")
        break

# ----------------------
# Save final result
# ----------------------
result_img = img_clamped.detach().cpu().squeeze(0)
result_img = ((result_img.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
Image.fromarray(result_img).save("adv_bob_face_only.png")

plt.imshow(result_img)
plt.axis('off')
plt.show()