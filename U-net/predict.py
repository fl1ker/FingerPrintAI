import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt  # Для визуализации

# Определение модели (то же, что в твоём коде)
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

# Устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = UNetWithResNet().to(device)
model.load_state_dict(torch.load("D:/Python/U-net/unet_fingerprint.pth"))
model.eval()  # Переводим модель в режим оценки (без обучения)

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Чёрно-белое изображение
    img = img.resize((96, 103), Image.BILINEAR)
    img = np.array(img) / 255.0  # Нормализация [0, 1]
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 103]
    if img.shape[-2:] == (103, 96):
        img = img.transpose(-2, -1)  # Меняем местами H и W: [1, 1, 96, 103]
    return img

# Путь к новой фотографии
image_path = "D:/Python/datasetFingerPrint/test_model/1__M_Left_index_finger.BMP"  # Укажи путь к своему изображению
output_dir = "D:/Python/datasetFingerPrint/predicted_masks"
os.makedirs(output_dir, exist_ok=True)

# Предобработка изображения
img_tensor = preprocess_image(image_path).to(device)

# Предсказание маски
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.sigmoid(output) > 0.5  # Бинаризация: вероятности в 0 или 1
    pred_mask = pred_mask.squeeze().cpu().numpy()  # Преобразуем в numpy массив

# Сохранение предсказанной маски
pred_mask = (pred_mask * 255).astype(np.uint8)  # Маска в формате 0 и 255
output_path = os.path.join(output_dir, "predicted_mask.bmp")
Image.fromarray(pred_mask).save(output_path)

# Визуализация (опционально)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original image")
plt.imshow(np.array(Image.open(image_path).convert("L")), cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
rotated_flipped_mask = np.fliplr(np.rot90(pred_mask, k=3))
plt.imshow(rotated_flipped_mask, cmap="gray")
plt.axis("off")
plt.show()

print(f"Предсказанная маска сохранена по пути: {output_path}")