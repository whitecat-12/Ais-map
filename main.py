import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from safetensors.torch import save_file  # Import tambahan untuk ekstensi safetensor

# 1. di dalam baris kode ini system akan mengambil data dari json
with open('waypoint.json', 'r') as f:
    data = json.load(f)

path_coordinates = []
for point in data["waypoints"]:
    path_coordinates.append([point["lat"], point["lng"]])

coords = torch.tensor(path_coordinates, dtype=torch.float32)

# 2. di dalam bari code ini kode akan menormalisasi data yang dibutuhkan termasuk rules atau data yang tertulis
coords_min = coords.min(dim=0)[0]
coords_max = coords.max(dim=0)[0]
coords_scaled = (coords - coords_min) / (coords_max - coords_min)

inputs = coords_scaled[:-1]
targets = coords_scaled[1:]


# 3. di kode ini Arsitektur Model kita menggunakan: MLP (Multi-Layer Perceptron) yang ditingkatkan dengan lstm
class RoutePredictor(nn.Module):
    def __init__(self):
        super(RoutePredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.layer(x)

model = RoutePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. dikode ini data akan ditraining sesuai aristektur mengguunakan MSELoss (Mean Squared Error),dan Adam Optimizer  
print(f"Memulai training dengan {len(coords)} titik dari waypoint.json...")
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.6f}')

print("Training selesai!")

# 5. dikode ini hasil akan disimpan dalam sistem safetensor
# Kita ambil bobot model (state_dict) lalu simpan
weights = model.state_dict()
save_file(weights, "model_rute.safetensors")

print("Model telah disimpan dalam format biner: 'model_rute.safetensors'")
