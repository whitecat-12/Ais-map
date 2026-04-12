import torch
import torch.nn as nn
import json
import os
from safetensors.torch import load_file

# 1. Arsitektur Model (Harus sama)
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

# 2. Persiapan Awal
print("--- Memuat Model AI... ---")
model = RoutePredictor()
try:
    weights = load_file("model_rute.safetensors")
    model.load_state_dict(weights)
    model.eval()
    
    with open('waypoint.json', 'r') as f:
        data = json.load(f)
    
    coords = torch.tensor([[p["lat"], p["lng"]] for p in data["waypoints"]], dtype=torch.float32)
    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    print("Model Berhasil Dimuat!\n")
except Exception as e:
    print(f"Error: Pastikan file .safetensors dan .json ada di folder ini. ({e})")
    exit()

# 3. Loop Input CMD
print("Ketik 'exit' untuk keluar.")
while True:
    input_user = input("Masukkan koordinat (format: lat, lng) contoh -6.1, 106.8: ")
    
    if input_user.lower() == 'exit':
        break
    
    try:
        # Pisahkan input berdasarkan koma
        lat_str, lng_str = input_user.split(',')
        lat_val = float(lat_str.strip())
        lng_val = float(lng_str.strip())

        # Normalisasi
        input_coord = torch.tensor([lat_val, lng_val], dtype=torch.float32)
        input_scaled = (input_coord - coords_min) / (coords_max - coords_min)
        
        # Prediksi
        with torch.no_grad():
            prediction_scaled = model(input_scaled)
        
        # Denormalisasi
        hasil = prediction_scaled * (coords_max - coords_min) + coords_min
        
        print(f"\n>>> HASIL PREDIKSI AI:")
        print(f"Latitude  : {hasil[0].item():.6f}")
        print(f"Longitude : {hasil[1].item():.6f}")
        print("-" * 30)

    except ValueError:
        print("Format salah! Gunakan angka dan pisahkan dengan koma (contoh: -6.1, 106.2)")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
