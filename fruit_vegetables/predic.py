import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm

# Definisi ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=False)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 36)  # Change output layer to match number of classes
model.load_state_dict(torch.load('vit_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as per ViT requirements
])

# Process image yang ingin di deteksi
image_path = 'img.png'
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)

# Membuat Prediksi
with torch.no_grad():
    output = model(image_tensor)

predicted_class = torch.argmax(output, dim=1).item()
print("Predicted class:", predicted_class)

# Hasil prediksi
plt.imshow(image)
plt.title(f"Predicted class: {predicted_class}")
plt.show()
