import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os

# Блок двойной свёртки
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# U-Net с ResNet18 для grayscale 96x103
class UNetWithResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetWithResNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.enc1_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            pretrained_weights = resnet.conv1.weight.mean(dim=1, keepdim=True)
            self.enc1_conv.weight.copy_(pretrained_weights)
        self.enc1 = nn.Sequential(self.enc1_conv, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.bottleneck = resnet.layer4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        e1 = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        out = F.interpolate(d1, size=(96, 103), mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return out

# Датасет для отпечатков пальцев
class FingerprintDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".BMP")]
        self.mask_paths = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir) if img.endswith(".bmp")]
        self.image_paths.sort()
        self.mask_paths.sort()
        self.transform = transform
        assert len(self.image_paths) == len(self.mask_paths), "Количество изображений и масок должно совпадать!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загрузка изображения
        img = Image.open(self.image_paths[idx]).convert("L")
        img = img.resize((96, 103), Image.BILINEAR)
        img = np.array(img) / 255.0  # Нормализация [0, 1]
        img = torch.FloatTensor(img).unsqueeze(0)  # [1, 96, 103]

        # Загрузка маски
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((96, 103), Image.BILINEAR)
        mask = np.array(mask) / 255.0  # Нормализация [0, 1]
        mask = torch.FloatTensor(mask).unsqueeze(0).round()  # [1, 96, 103], 0 или 1

        # Исправление ориентации (транспонирование, если нужно)
        if img.shape[-2:] == (103, 96):  # Если размер [1, 103, 96]
            img = img.transpose(-2, -1)  # Меняем местами H и W: [1, 96, 103]
        if mask.shape[-2:] == (103, 96):  # Если размер [1, 103, 96]
            mask = mask.transpose(-2, -1)  # Меняем местами H и W: [1, 96, 103]

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# Подготовка данных
image_dir = "D:/Python/datasetFingerPrint/Real"
mask_dir = "D:/Python/datasetFingerPrint/Real_Mask"
dataset = FingerprintDataset(image_dir, mask_dir)

# Отладочный вывод
print(f"Found {len(dataset.image_paths)} images in {image_dir}")
print(f"Found {len(dataset.mask_paths)} masks in {mask_dir}")
print(f"Total dataset size: {len(dataset)}")

if len(dataset) == 0:
    raise ValueError("Датасет пуст! Проверь пути и наличие файлов.")

# Разделение данных
total_size = len(dataset)  # 6000
train_size = int(0.7 * total_size)  # 4200
test_size = int(0.2 * total_size)   # 1200
val_size = total_size - train_size - test_size  # 600
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Даталоадеры
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Тренировка
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

model = UNetWithResNet().to(device)
model.load_state_dict(torch.load("D:/Python/U-net/unet_fingerprint.pth"))

"""
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30  # Уменьшено для теста
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:  # Вывод каждые 100 итераций
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Сохранение модели
torch.save(model.state_dict(), "D:/Python/U-net/unet_fingerprint.pth")
"""

# Оценка (пункт g)
def dice_score(pred, target):
    pred = torch.sigmoid(pred) > 0.5  # Бинаризация
    target = target.bool()
    intersection = (pred & target).sum().float()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    target = target.bool()
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return intersection / (union + 1e-8)

model.eval()
dice_total, iou_total = 0, 0
with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        dice_total += dice_score(outputs, masks).item()
        iou_total += iou_score(outputs, masks).item()

print(f"Average Dice: {dice_total/len(val_loader):.4f}")
print(f"Average IoU: {iou_total/len(val_loader):.4f}")