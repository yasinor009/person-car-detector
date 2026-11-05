# Laptop preview models

- `person_car_yolo26n.onnx`: COCO general detector, dynamic raw export; CPU profile uses 640px.
- `person_car_yolo26s.onnx`: COCO general detector, dynamic raw export; balanced 960px, sliced long-range 1280px.
- `visdrone11s_1280.onnx`: publisher-trained aerial VisDrone model; static batch-one 1280px.
- `person_car_yolo11n.onnx`: legacy v3 pipeline comparison; static 640px.

Only canonical person/car results are emitted. Internal class sets are COCO80 or
VisDrone10, not a claim of custom two-class training. Published model quality does
not certify >=90% precision/recall on the user's camera or detection at 150m.

Exact hashes/sources/contracts: `PROVENANCE_V4.json`. Verify from project root with
`python verify_models.py`. Optional larger aerial model:
`python download_optional_model.py` (exact tested hash required).

Upstream Ultralytics-derived weights use AGPL-3.0 or an appropriate commercial
license. See `ULTRALYTICS_LICENSE.txt` and upstream source/model cards. Publisher
model-card/validation material and the VisDrone dataset can have separate terms.
The archive does not relicense upstream weights or bundle dataset images.
