import torch
import torch.nn as nn
import json
import os
from safetensors.torch import load_file

# 1. Arsitektur Model 
class RoutePredictor(nn.Module):
    def __init__(self):
        super(RoutePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=False)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 2. Inisialisasi dan Load Weights
print("--- Memuat AI Navigasi (Hybrid LSTM) ---")
model = RoutePredictor()

if not os.path.exists("model_rute.safetensors"):
    print("Error: File 'model_rute.safetensors' tidak ditemukan! Silahkan training dulu.")
    exit()

# Load model
weights = load_file("model_rute.safetensors")
model.load_state_dict(weights)
model.eval()

# Load data waypoint untuk kebutuhan Min-Max Normalisasi
with open('waypoint.json', 'r') as f:
    data = json.load(f)

path_coordinates = [[p["lat"], p["lng"]] for p in data["waypoints"]]
coords = torch.tensor(path_coordinates, dtype=torch.float32)

# Ambil nilai min/max yang sama persis dengan saat training
coords_min = coords.min(dim=0, keepdim=True)[0]
coords_max = coords.max(dim=0, keepdim=True)[0]

print("Sistem Siap! AI sudah hafal pola waypoint-mu.\n")

# 3. Loop Prediksi Interaktif
while True:
    try:
        input_user = input("Masukkan koordinat SEKARANG (lat, lng) atau 'exit': ")
        if input_user.lower() == 'exit': break
        
        # Parsing input
        lat_start, lng_start = map(float, input_user.split(','))
        
        # Berapa langkah rute yang mau dibuat?
        steps = input("Mau prediksi berapa titik ke depan? (default 10): ")
        steps = int(steps) if steps.strip() != "" else 10

        curr_lat, curr_lng = lat_start, lng_start

        print(f"\n[ HASIL PREDIKSI RUTE ]")
        print(f"{'No':<5} | {'Latitude':<12} | {'Longitude':<12}")
        print("-" * 35)

        for i in range(1, steps + 1):
            # A. Normalisasi input
            current_tensor = torch.tensor([[curr_lat, curr_lng]], dtype=torch.float32)
            scaled_input = (current_tensor - coords_min) / (coords_max - coords_min + 1e-8)
            
            # B. Siapkan dimensi untuk LSTM: (Sequence=1, Batch=1, Feature=2)
            lstm_input = scaled_input.unsqueeze(0)
            
            # C. Prediksi
            with torch.no_grad():
                scaled_output = model(lstm_input)
            
            # D. Denormalisasi (Kembalikan ke koordinat peta)
            # scaled_output shape: (1, 1, 2) -> ambil [0,0]
            result = scaled_output[0, 0] * (coords_max - coords_min + 1e-8) + coords_min
            
            final_lat = result[0, 0].item()
            final_lng = result[0, 1].item()

            print(f"{i:<5} | {final_lat:<12.6f} | {final_lng:<12.6f}")

            # Update posisi sekarang untuk langkah berikutnya (Auto-regressive)
            curr_lat, curr_lng = final_lat, final_lng

        print("-" * 35)
        print("Prediksi selesai.\n")

    except ValueError:
        print("Format salah! Gunakan: lat, lng (contoh: -7.2, 112.7)")
    except Exception as e:
        print(f"Terjadi error: {e}")
