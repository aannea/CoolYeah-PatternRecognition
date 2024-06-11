import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Mendefinisikan ulang kelas SimpleNet yang sama
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Memeriksa apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Memuat model yang telah disimpan
model = SimpleNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.to(device)
model.eval()

# Transformasi untuk gambar baru
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Memuat dan memproses gambar baru
image_path = ('img_9.png')
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0).to(device)

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
