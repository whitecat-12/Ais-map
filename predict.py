import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from safetensors.torch import save_file

# 1. Loading Data dengan proteksi jika file tidak ada
if not os.path.exists('waypoint.json'):
    print("Error: File 'waypoint.json' tidak ditemukan!")
    # Contoh data dummy agar kode tetap jalan untuk tes
    data = {"waypoints": [{"lat": -6.2, "lng": 106.8}, {"lat": -6.21, "lng": 106.81}]}
else:
    with open('waypoint.json', 'r') as f:
        data = json.load(f)

path_coordinates = [[point["lat"], point["lng"]] for point in data["waypoints"]]
coords = torch.tensor(path_coordinates, dtype=torch.float32)

# 2. Normalisasi (Ditambah epsilon 1e-8 agar tidak pembagian nol)
coords_min = coords.min(dim=0, keepdim=True)[0]
coords_max = coords.max(dim=0, keepdim=True)[0]
# Epsilon mencegah error jika coords_max == coords_min
coords_scaled = (coords - coords_min) / (coords_max - coords_min + 1e-8)

# Input: titik sekarang, Target: titik berikutnya
# Untuk LSTM, kita butuh dimensi: (Sequence, Batch, Feature)
inputs = coords_scaled[:-1].unsqueeze(1) 
targets = coords_scaled[1:].unsqueeze(1)

# 3. Arsitektur Model: Hybrid MLP + LSTM
class RoutePredictor(nn.Module):
    def __init__(self):
        super(RoutePredictor, self).__init__()
        # Menggunakan LSTM agar benar-benar "mengingat" urutan rute
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=False)
        self.fc = nn.Linear(64, 2) # Output layer kembali ke koordinat lat/lng
        
    def forward(self, x):
        # x shape: (seq_len, batch, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = RoutePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
print(f"Memulai training dengan {len(coords)} titik...")
model.train()

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Cek jika loss NaN (biasanya karena data input rusak)
    if torch.isnan(loss):
        print(f"Training terhenti di epoch {epoch} karena Loss NaN!")
        break
        
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.6f}')

print("Training selesai!")

# 5. Simpan ke Safetensors
model.eval()
weights = model.state_dict()
save_file(weights, "model_rute.safetensors")

print("Model telah disimpan: 'model_rute.safetensors'")
