import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = torch.load('simple_cnn.pth')
model.eval()  # Pastikan model dalam mode evaluasi

# Transformasi untuk gambar baru
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Memuat dan memproses gambar baru
image_path = 'img.png'
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)

# Membuat prediksi
with torch.no_grad():
    output = model(image_tensor)

# Mendapatkan kelas yang diprediksi
_, predicted_class = torch.max(output, 1)
print("Predicted class:", predicted_class.item())

# Menampilkan gambar dan prediksi
plt.imshow(image, cmap='gray')
plt.title(f"Predicted class: {predicted_class.item()}")
plt.show()
