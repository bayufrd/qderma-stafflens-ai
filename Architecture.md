# StaffLens AI Architecture

## Prinsip

Kamera/inference **tidak pernah menunggu Vision AI**. Vision AI hanya fallback async.

```text
camera frame
  -> YOLOX (Apache-2.0) person detection
  -> ByteTrack
  -> snapshot terbaik ke customer_snapshots/temp
  -> local face/body analysis
  -> observasi multi-frame + dwell
  -> customer provisional / staff
  -> async Vision AI (opsional)
  -> verified / reclassified / review
  -> promote atau cleanup snapshot
```

## Lifecycle snapshot

- `temp/`: snapshot crop awal/terbaik; bukan bukti final.
- Satu sampai tiga snapshot berkualitas per track; jangan simpan tiap frame.
- Kualitas: ukuran bbox, ketajaman, brightness, dan crop tidak terlalu terpotong.
- Customer dapat dicatat lokal setelah observasi selesai; Vision AI boleh mengoreksi kemudian.
- Hasil staff dari Vision AI melakukan rollback customer provisional.
- Cleanup temp wajib berdasarkan umur; kegagalan storage tidak boleh menghentikan loop kamera.

## Timeout dan queue

- Worker Vision AI terpisah dari inference.
- Request memiliki timeout terbatas; respons lambat menjadi `unverified`, bukan bottleneck kamera.
- Batasi concurrency/request agar provider lambat tidak membuat thread tak terbatas.
- Simpan `visitor_id`/snapshot path, bukan hanya `track_id`; track dapat hilang sebelum respons datang.
- Retry terbatas; hasil gagal tidak menghapus customer lokal.

## UI/API

- Preview/status kamera: WebSocket realtime.
- Dashboard/table: polling sekitar 3 detik.
- Resource: polling sekitar 5 detik.
- Status sebaiknya menampilkan `observing`, `provisional`, `verified`, `reclassified`, `unverified`.

## Model

- Produksi: `FACE_MODEL=sface` (OpenCV Zoo YuNet + SFace, Apache-2.0, 128-D embedding).
- Setelah mengganti model wajah, jalankan `python reembed_faces.py` untuk memperbarui embedding seluruh foto pegawai.
- YOLOX detection (`yolox_nano.onnx`, Apache-2.0) bukan keputusan customer. Tangan/jam/krop kecil wajib ditolak lewat filter ukuran, confidence, aspect ratio, serta konfirmasi beberapa frame.

## Konfigurasi baseline

```env
FACE_MODEL=sface
CUSTOMER_MIN_DWELL_SECONDS=2.5
VISION_AI_TIMEOUT_SECONDS=30
CUSTOMER_SNAPSHOT_DIR=customer_snapshots/temp
```

Jangan commit `.env`, API key, password, atau URL DB bercredential. Jika pernah terekspos, segera rotate credential.
