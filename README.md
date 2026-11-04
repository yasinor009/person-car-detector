# Person / car detection — 4.0 laptop preview

**The requested >=90% precision AND recall for both classes, including measured
>=150 m targets, has NOT been reached or certified.** See
[ACCEPTANCE_STATUS.json](ACCEPTANCE_STATUS.json) and the actual benchmark results.

راهنمای کامل فارسی و نصب ویندوز/لینوکس: [README_LAPTOP_FA.md](README_LAPTOP_FA.md).
نتایج واقعی: [LAPTOP_EXPERIMENTS_FA.md](LAPTOP_EXPERIMENTS_FA.md).
مقالات، منابع و دلایل انتخاب: [LAPTOP_RESEARCH_FA.md](LAPTOP_RESEARCH_FA.md).

Windows: run `install_laptop.bat`, then `start_laptop.bat`; open
http://127.0.0.1:5000 in a browser on that laptop. NVIDIA setup is optional and
requires a supported GPU/driver; use `install_nvidia.bat` only for NVIDIA.

Linux:
```bash
bash install_laptop.sh
.venv/bin/python run_laptop.py --profile balanced
```

The new runtime preserves source resolution, supports CPU/CUDA ONNX and optional
PyTorch inference, raw/end-to-end model contracts, class-name mappings for COCO,
VisDrone and custom two-class exports, native-pixel overlapping slices, dynamic
batching where exported, and a latest-frame camera queue. Only person/car results
are emitted. Predicted/weak boxes are not shown as confirmed detections.

Bundled models: YOLO26n, YOLO26s, aerial-trained YOLO11s/1280, and the legacy
YOLO11n for comparisons. A larger aerial YOLO26m is optional. These are published
pretrained models, not weights trained on user footage in this session. Upstream
model licensing and exact tested hashes are recorded in
[models/PROVENANCE_V4.json](models/PROVENANCE_V4.json).

`main.py` and v3 documentation remain for legacy reference. For this laptop
release use `run_laptop.py`; for the prior phone release use the separate v3.1 ZIP.
