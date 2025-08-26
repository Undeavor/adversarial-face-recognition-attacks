import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# Load FaceNet
# ---------------------------
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------------------------
# Load target face and compute embedding
# ---------------------------
target_face_path = 'bob.jpg'
face_img = Image.open(target_face_path).convert('RGB').resize((160,160))
face_tensor = torch.from_numpy(np.array(face_img)).float() / 127.5 - 1
face_tensor = face_tensor.permute(2,0,1).unsqueeze(0).to(device)
target_embedding = model(face_tensor).detach()
target_embedding = target_embedding / target_embedding.norm(dim=1, keepdim=True)
torch.save(target_embedding, 'target_embedding.pt')
print("Target embedding saved.")

# ---------------------------
# Initialize random noise image
# ---------------------------
adv_img = torch.randn(1,3,160,160,device=device, requires_grad=True)
optimizer = torch.optim.Adam([adv_img], lr=0.05)
cos = torch.nn.CosineSimilarity(dim=1)
max_steps = 2000
similarity_threshold = 0.95

# ---------------------------
# Optimization loop
# ---------------------------
for step in range(max_steps):
    optimizer.zero_grad()
    
    img_norm = torch.clamp(adv_img, -1, 1)
    embedding = model(img_norm)
    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    
    loss = 1 - (embedding * target_embedding).sum(dim=1).mean()
    loss.backward()
    
    optimizer.step()
    
    if step % 100 == 0:
        similarity = (embedding * target_embedding).sum(dim=1).item()
        print(f"Step {step} - Loss: {loss.item():.6f} - Cosine similarity: {similarity:.4f}")
        if similarity >= similarity_threshold:
            print(f"✅ Target face recognized at step {step}!")
            break

# ---------------------------
# Save adversarial image
# ---------------------------
result_img = ((torch.clamp(adv_img, -1,1).detach().cpu().squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).astype(np.uint8)
Image.fromarray(result_img).save('adversarial_noise_face.png')
print("Adversarial noise face saved!")

# ---------------------------
# Display
# ---------------------------
plt.imshow(result_img)
plt.axis('off')
plt.show()

# ---------------------------
# Verification
# ---------------------------
embedding = model(torch.clamp(adv_img, -1,1))
embedding = embedding / embedding.norm(dim=1, keepdim=True)
similarity = (embedding * target_embedding).sum(dim=1).item()
print(f"Final Cosine similarity: {similarity:.4f}")