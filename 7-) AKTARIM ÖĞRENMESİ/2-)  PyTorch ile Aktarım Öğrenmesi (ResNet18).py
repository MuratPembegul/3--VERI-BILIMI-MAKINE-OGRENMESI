import torch
import torch.nn as nn
import torchvision.models as models

# Önceden eğitilmiş ResNet18 modelini yükleyelim
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features

# Yeni sınıflandırıcı katmanı ekleyelim
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),  # 10 sınıflı bir problem için
)

# Modeli GPU'ya taşıyalım
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizasyon ve loss fonksiyonu
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
