# مدل‌های فعلی

- `person_car_yolo11n.onnx`: مدل سبک پیش‌فرض Termux، ورودی ثابت ۶۴۰.
- `person_car_yolo26n.onnx`: مدل عمومی CPU برای پروفایل `cpu`.
- `person_car_yolo26s.onnx`: مدل عمومی بزرگ‌تر برای `balanced` و `long-range`.

برنامه فقط خروجی‌های canonical با نام `person` و `car` را منتشر می‌کند.
مدل‌ها روی COCO آموزش عمومی دیده‌اند و مخصوص دوربین یا خودروی کاربر نیستند.
کارایی روی دوربین مقصد و فاصله‌های دور باید با داده برچسب‌دار اندازه‌گیری شود.

هش، منبع، قرارداد ورودی/خروجی و نسخه export در `PROVENANCE_V4.json` است.
بررسی فایل‌ها:

```bash
python verify_models.py
```

وزن‌های مشتق از Ultralytics تحت AGPL-3.0 یا مجوز تجاری مناسب upstream هستند.
جزئیات در `ULTRALYTICS_LICENSE.txt` ثبت شده است.
