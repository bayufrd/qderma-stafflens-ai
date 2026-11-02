# StaffLens AI

> Hybrid CCTV & Webcam Analytics untuk **Penghitungan Customer Unik Harian (*Unique Daily Customer Counting*)** dan **Pemantauan Kehadiran Staf (*Active Staff Presence*)**: **YOLOv8 + Supervision/ByteTrack** untuk deteksi dan pelacakan orang, dipadukan dengan **InsightFace (`buffalo_l`)**, **HSV Body Appearance Re-ID**, **Smart Fallback Vision AI**, **PostgreSQL**, serta **Dashboard Web Lokal (FastAPI + Vue 3)**.

---

## Ringkasan Tujuan & Fitur Utama

StaffLens AI dirancang untuk menjawab dua kebutuhan operasional utama di lokasi kerja/klinik:
1. **Menghitung Jumlah Customer Unik per Hari (`1 Hari = 1 Customer`):** Meskipun seorang customer keluar-masuk kamera atau berkunjung 2–3 kali di hari yang sama, sistem mengenalinya kembali dan tetap menghitungnya sebagai **1 customer unik** (hanya menambah frekuensi kunjungan ulang).
2. **Menghitung Karyawan yang Hadir / Aktif & Mengecualikannya dari Customer:** Karyawan yang sudah didaftarkan wajahnya dihitung pada metrik **Karyawan Hadir Hari Ini** dan **tidak pernah dihitung sebagai customer**.

### Kapabilitas Inti (`app.py` + `server.py`)
- **Deteksi & Tracking (`YOLOv8n` + `ByteTrack`):** Mendeteksi orang (`COCO class 0`) di setiap frame dan menjaga `track_id` tetap stabil.
- **3 Lapis Deduplikasi Pengunjung Harian (*Ephemeral Daily Visitor Memory*):**
  1. **Auto-Enroll Wajah Pengunjung Harian (`face_vector`):** Pengunjung yang wajahnya terlihat (dan bukan karyawan) disimpan vektor wajah 512-D-nya ke tabel `daily_visitors` untuk tanggal hari tersebut. Jika ia datang lagi di hari yang sama, wajahnya dikenali sehingga tidak dihitung ganda.
  2. **Body Appearance Signature (`body_vector`):** Untuk pengunjung atau staf yang tidak terlihat jelas wajahnya, sistem mengekstrak *signature* warna dan kontras pakaian atas/bawah (*HSV Torso Histogram*). Ini mencegah orang yang sama dihitung 2 kali saat kembali ke area kamera di hari yang sama.
  3. **Grace Period (`2.5 detik`) & *Retroactive Employee Rollback*:** Subjek baru diamati terlebih dahulu sebelum dicatat sebagai customer. Jika seorang karyawan masuk membelakangi kamera (sempat tercatat sementara sebagai pengunjung) lalu menoleh dan wajahnya dikenali sebagai karyawan, sistem otomatis **membatalkan (*rollback*)** catatan customer tersebut dan memindahkannya ke kehadiran karyawan.
- **Smart Fallback Vision AI (Opsional via `.env`):** Jika `VISION_AI_API_KEY` diisi, sistem mengirim **1 snapshot crop tubuh secara asinkron (non-blocking)** saat skor kemiripan wajah "nanggung" (*gray zone* `0.30–0.44`, misalnya tertutup masker) atau saat wajah tidak terlihat, guna memverifikasi apakah subjek mengenakan atribut/seragam staf klinik atau berpakaian customer.
- **Portal Dashboard Lokal (`127.0.0.1`):** Antarmuka Vue 3 + FastAPI dengan metrik **Customer Unik Hari Ini**, **Karyawan Hadir Hari Ini**, **Subjek di Kamera Saat Ini**, tabel registri customer harian beserta jumlah kunjungan ulangnya, status kehadiran per pegawai, *live stream* MJPEG, serta pengaturan performa kamera (Level 1–5).

---

## Arsitektur & Alur Pipeline

```text
Webcam (Indeks 0..8) / RTSP / Video File
          │
          ▼
   OpenCV Frame Capture (app.py)
   [Auto-refresh data pegawai & daily_visitors dari PostgreSQL]
          │
          ▼
   YOLOv8 Person Detection (class=0) + Supervision ByteTrack (track_id)
          │
          ├── 1. Cek Wajah Selektif (InsightFace buffalo_l 512-D)
          │        ├── Cocok dengan DB Karyawan (>= 0.45)
          │        │     └──► Kunci sebagai KARYAWAN (Kotak Hijau)
          │        │          + Catat kehadiran staf hari ini
          │        │          + Rollback otomatis jika sebelumnya sempat tercatat Customer
          │        └── Skor Nanggung (0.30 - 0.44)
          │              └──► Simpan kandidat untuk verifikasi Fallback Vision AI
          │
          └── 2. Setelah Masa Observasi (CUSTOMER_MIN_DWELL_SECONDS = 2.5s)
                   ├── Cocok dengan Body Signature Staf Aktif Hari Ini
                   │     └──► Kunci sebagai KARYAWAN (Staf Body Re-ID)
                   │
                   └── Bukan Karyawan ──► Cek Memori Pengunjung Hari Ini (daily_visitors)
                         ├── Cocok Wajah (>= 0.42) ATAU Pola Pakaian (>= 0.84)
                         │     └──► CUSTOMER LAMA HARI INI (Tidak +1 Customer Unik, tambah visit_count)
                         │
                         └── Belum Ada di Memori Hari Ini
                               └──► CUSTOMER UNIK BARU (+1 Customer Hari Ini, Kotak Biru)
                                    + Opsional: Verifikasi Seragam/Atribut via Vision AI di background
```

---

## Penjelasan Flow Foto Lokal (`db_karyawan/`) vs Vector Embedding PostgreSQL

1. **Apakah Ada Proses "Training Model" Saat Mendaftarkan Foto?**
   - **Tidak ada retraining.** Model `InsightFace (buffalo_l)` adalah model *pre-trained* (*Zero-Shot Metric Learning*). Saat Anda menjalankan `register_employee.py`, foto wajah pegawai diubah menjadi **512 angka desimal (vektor 512-dimensi)** dan disimpan ke kolom `vector` (`JSONB`) pada tabel `face_embeddings` di PostgreSQL.
2. **Apakah Foto di `db_karyawan/` Masih Dibutuhkan Setelah Masuk PostgreSQL?**
   - Saat kamera berjalan (`app.py` / `server.py`), aplikasi **hanya membaca vektor 512 angka dari PostgreSQL** dan **tidak membaca folder `db_karyawan/` sama sekali**.
   - Namun, disarankan tetap menyimpan cadangan foto asli (di folder lokal yang di-ignore Git atau di S3 privat) apabila suatu saat Anda ingin menambah foto baru untuk pegawai tersebut atau mengganti jenis model `FACE_MODEL` di `.env`.

---

## Struktur Proyek

```text
qderma-stafflens-ai/
├── app.py                  # Pipeline utama CV/AI: deteksi, deduplikasi customer harian, & kehadiran staf
├── server.py               # Backend FastAPI, API metrik harian (/api/overview, /api/visitors), & MJPEG stream
├── database.py             # Model SQLAlchemy (Employee, FaceEmbedding, DailyVisitor, CameraEvent, CameraSettings)
├── register_employee.py    # CLI ekstraksi foto wajah pegawai menjadi vektor 512-D di PostgreSQL
├── requirements.txt        # Daftar dependensi Python
├── .env.example            # Template konfigurasi environment (DB, Performa, Deduplikasi, & Vision AI)
├── yolov8n.pt              # Bobot model dasar YOLOv8 Nano (COCO)
├── db_karyawan/            # Folder lokal foto referensi pegawai (diabaikan oleh Git)
├── db_customer/             # Maks. 10 snapshot crop terbaik per ID customer harian (diabaikan oleh Git)
├── frontend/               # Aplikasi SPA Dashboard (Vue 3 + Vite)
│   ├── src/App.vue         # UI Ringkasan Customer/Staf, Registri Customer Harian, Pegawai, & Pengaturan
│   └── dist/               # Hasil build produksi yang disajikan oleh server.py
└── docs/                   # Dokumentasi panduan operasional & konfigurasi
```

---

## Snapshot Customer Lokal

- Overlay live menampilkan identitas di atas kotak saja: customer `CUST-001`, karyawan `K-001-Bayu`, atau status `Observasi...`; FPS/hitungan agregat tidak ditulis menimpa frame.
- Saat kamera berjalan, sistem mengambil crop track customer berkala dan menilai ketajaman (Laplacian); hanya crop valid/tidak terlalu blur yang disimpan.
- Maksimal **10 snapshot dengan skor ketajaman tertinggi** per ID customer harian di `db_customer/<CUST-YYYYMMDD-NNN>/`. Snapshot tersedia melalui galeri saat row customer diklik.
- ID customer berlaku untuk registri harian (`daily_visitors`); kunjungan hari lain tidak otomatis digabung. Crop dapat menampilkan wajah maupun pakaian/tubuh; data ini berpotensi memuat informasi pribadi.
- Gambar hanya disajikan lewat API yang memerlukan autentikasi. Folder `db_customer/` di-ignore Git; file lokal tidak ikut backup kecuali dikonfigurasi terpisah. Hapus folder/file secara lokal untuk menghapus snapshot. Jika customer kemudian direklasifikasi sebagai staf, snapshot provisional dihapus otomatis.

## Skema Database (`database.py`)

| Nama Tabel | Model ORM | Fungsi & Kolom Utama |
|---|---|---|
| `employees` | `Employee` | Profil pegawai terdaftar (`id`, `employee_code`, `name`, `role`, `active`, `created_at`). |
| `face_embeddings` | `FaceEmbedding` | Vektor wajah 512-D pegawai (`id`, `employee_id`, `model_name`, `vector` `JSONB`, `quality_score`). |
| `daily_visitors` | `DailyVisitor` | **Registri Customer Unik Harian** (`id`, `visit_date`, `visitor_code`, `face_vector`, `body_vector`, `attributes_summary`, `visit_count`, `first_seen_at`, `last_seen_at`, `reclassified_employee_id`). |
| `camera_events` | `CameraEvent` | Log aktivitas deteksi staf (`identity_locked`) maupun kedatangan customer (`customer_new`, `customer_return`). |
| `camera_settings` | `CameraSettings` | Pengaturan indeks kamera, level performa (1–5), dan resolusi dari Portal Dashboard. |
| `audit_logs` | `AuditLog` | Tabel log audit sistem. |

---

## Cara Penggunaan (Quick Start)

### 1. Persiapan Environment & Database

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Pastikan `.env` berisi `DATABASE_URL` PostgreSQL dan `DASHBOARD_PASSWORD`, lalu inisialisasi tabel database:

```bash
.venv/bin/python database.py
```

### 2. Daftarkan Wajah Karyawan (Agar Staf Dikecualikan dari Hitungan Customer)

Masukkan 3–5 foto wajah staf (1 wajah per foto) ke dalam `db_karyawan/<nama>/`, lalu jalankan:

```bash
.venv/bin/python register_employee.py \
  --code EMP-001 \
  --name "Bayu" \
  --role "Karyawan" \
  --photos db_karyawan/bayu
```

### 3. Jalankan Portal Dashboard Web

```bash
npm install --prefix frontend
npm run build --prefix frontend
.venv/bin/python server.py
```

Buka `http://127.0.0.1:8000`, masuk dengan `DASHBOARD_PASSWORD`, lalu klik **Mulai kamera & AI**.

---

## Konfigurasi Deduplikasi & Smart Fallback Vision AI (`.env`)

| Variabel `.env` | Nilai Default | Fungsi |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.45` | Ambang kemiripan wajah untuk mengunci identitas karyawan. |
| `VISITOR_FACE_THRESHOLD` | `0.42` | Ambang kemiripan wajah untuk mengenali customer yang balik lagi di hari yang sama. |
| `BODY_MATCH_THRESHOLD` | `0.84` | Ambang kemiripan *HSV Torso Signature* (pola warna pakaian atas/bawah) di hari yang sama. |
| `CUSTOMER_MIN_DWELL_SECONDS` | `2.5` | Durasi observasi minimum (detik) sebelum subjek baru dihitung sebagai customer. |
| `VISITOR_REENTRY_GAP_SECONDS` | `30` | Jeda waktu minimum (detik) sebelum kemunculan kembali customer yang sama dihitung sebagai kunjungan ulang (`visit_count + 1`). |
| `VISION_AI_API_KEY` | *(kosong)* | Opsional: isi dengan API Key (OpenAI-compatible) untuk mengaktifkan verifikasi seragam/atribut otomatis pada kasus skor wajah nanggung (*gray zone*) atau membelakangi kamera. |
| `VISION_AI_BASE_URL` | `https://api.openai.com/v1/chat/completions` | Endpoint Vision AI (mendukung OpenAI, Gemini OpenAI-compatible, OpenRouter, dll.). |
| `VISION_AI_MODEL` | `gpt-4o-mini` | Nama model Vision AI yang digunakan saat fallback. |
| `VISION_AI_TIMEOUT_SECONDS` | `120` | Timeout req Vision AI dalam detik; dijepit ke rentang 120–300 dtk (2–5 menit). |

---

## Dokumentasi Lengkap

- [Panduan Portal Dashboard Lokal](docs/DASHBOARD.md)
- [Panduan Webcam, Pendaftaran Pegawai, dan Database](docs/WEBCAM_DAN_DATABASE.md)
- [Setup Konfigurasi Infrastruktur](docs/SETUP_KONFIGURASI.md)
- [Integrasi SatuSehat](docs/INTEGRASI_SATUSEHAT.md)
- [Panduan Penggunaan Pipeline Dasar](docs/PANDUAN_PENGGUNAAN.md)
