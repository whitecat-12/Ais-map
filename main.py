import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from safetensors.torch import save_file

# 1. Load Data
try:
    with open('waypoint.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: File 'waypoint.json' tidak ditemukan!")
    exit()

path_coordinates = [[point["lat"], point["lng"]] for point in data["waypoints"]]
coords = torch.tensor(path_coordinates, dtype=torch.float32)

# 2. Normalisasi (Penting: Simpan nilai min/max untuk inferensi nanti)
coords_min = coords.min(dim=0)[0]
coords_max = coords.max(dim=0)[0]
# Hindari pembagian dengan nol jika data statis
coords_scaled = (coords - coords_min) / (coords_max - coords_min + 1e-8)

inputs = coords_scaled[:-1]
targets = coords_scaled[1:]

# 3. Arsitektur Model: MLP (Multi-Layer Perceptron)
class RoutePredictor(nn.Module):
    def __init__(self):
        super(RoutePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), # Menambah stabilitas training
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

model = RoutePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # LR sedikit dikecilkan agar tidak jumping

# 4. Training Loop
print(f"Memulai training dengan {len(inputs)} pasangan titik...")
model.train()

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.8f}')

print("Training selesai!")

# 5. Simpan Model & Metadata
# Safetensors hanya menyimpan tensor. Kita perlu simpan min/max agar bisa dipakai saat prediksi.
weights = model.state_dict()
# Tambahkan nilai normalisasi ke dalam file agar aplikasi lain tahu cara scaling-nya
weights_to_save = {k: v for k, v in weights.items()}
weights_to_save["coords_min"] = coords_min
weights_to_save["coords_max"] = coords_max

save_file(weights_to_save, "model_rute.safetensors")
print("Model dan Metadata normalisasi disimpan ke 'model_rute.safetensors'")
