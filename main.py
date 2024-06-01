#image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Memeriksa apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformasi untuk dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Memuat dataset MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Mendefinisikan jaringan neural sederhana dengan satu lapisan linear
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # Menggunakan satu lapisan linear

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc(x)
        return x

net = SimpleNet().to(device)

# Menggunakan DataParallel untuk memanfaatkan beberapa GPU
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)

net.to(device)

# Mendefinisikan loss function dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Melatih jaringan
for epoch in range(5):  # loop melalui dataset beberapa kali
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Mereset gradien optimizer
        optimizer.zero_grad()

        # Forward, backward, optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Menampilkan progres setiap 200 mini-batches
        if i % 200 == 199:  # print setiap 200 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}")
            running_loss = 0.0

print('Finished Training')

# Menguji jaringan dengan data uji
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
torch.save(net.state_dict(), 'mnist_model.pth')
model = SimpleNet()
model.load_state_dict(torch.load('mnist_model.pth'))