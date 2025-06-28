from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend jika sudah deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

