import torch
import torch.nn as nn
import json
import os
from safetensors.torch import load_file

# 1. Arsitektur Model (Tetap sama agar cocok dengan file .safetensors)
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

# 2. Persiapan Awal & Memuat Data
print("--- Memuat Model AI & Konfigurasi... ---")
model = RoutePredictor()

try:
    # Memuat bobot model
    weights = load_file("model_rute.safetensors")
    model.load_state_dict(weights)
    model.eval() # Mode prediksi (bukan training)
    
    # Memuat file JSON untuk referensi normalisasi (Min-Max)
    with open('waypoint.json', 'r') as f:
        data = json.load(f)
    
    coords = torch.tensor([[p["lat"], p["lng"]] for p in data["waypoints"]], dtype=torch.float32)
    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    print("Sistem Siap! Model dan data referensi berhasil dimuat.\n")

except Exception as e:
    print(f"ERROR: Pastikan file 'model_rute.safetensors' dan 'waypoint.json' ada di folder ini.")
    print(f"Detail Error: {e}")
    exit()

# 3. Loop Prediksi Beruntun (10 Titik)
print("Ketik 'exit' untuk keluar.")
while True:
    input_user = input("\nMasukkan Koordinat Awal (lat, lng) - Contoh: -6.1, 106.8: ")
    
    if input_user.lower() == 'exit':
        print("Program ditutup.")
        break
    
    try:
        # Parsing input user
        lat_str, lng_str = input_user.split(',')
        curr_lat = float(lat_str.strip())
        curr_lng = float(lng_str.strip())

        print(f"\n[ AI MENGHASILKAN 10 TITIK RUTE BERIKUTNYA ]")
        print(f"{'No':<5} | {'Latitude':<12} | {'Longitude':<12}")
        print("-" * 35)

        # Loop untuk menghasilkan 10 titik secara runtut
        for i in range(1, 11):
            # A. Normalisasi input saat ini
            input_coord = torch.tensor([curr_lat, curr_lng], dtype=torch.float32)
            input_scaled = (input_coord - coords_min) / (coords_max - coords_min)
            
            # B. Jalankan Prediksi
            with torch.no_grad():
                prediction_scaled = model(input_scaled)
            
            # C. Denormalisasi (Kembalikan ke angka koordinat asli)
            hasil = prediction_scaled * (coords_max - coords_min) + coords_min
            
            next_lat = hasil[0].item()
            next_lng = hasil[1].item()
            
            # D. Tampilkan hasil titik ke-i
            print(f"{i:<5} | {next_lat:<12.6f} | {next_lng:<12.6f}")
            
            # E. PENTING: Titik yang dihasilkan sekarang menjadi input untuk titik selanjutnya
            curr_lat = next_lat
            curr_lng = next_lng

        print("-" * 35)
        print("Selesai generate rute.")

    except ValueError:
        print("Format salah! Masukkan angka lat dan lng dipisah koma.")
    except Exception as e:
        print(f"Terjadi kesalahan teknis: {e}")
