# تشخیص انسان و خودرو و تخمین فاصله

پروژه Python برای تشخیص دو کلاس `person` و `car`، ردیابی مستقل اهداف و
فاصله‌سنجی هندسی کالیبره‌شده.

**وضعیت: پژوهشی.** دقت ۱۰۰ متر و اجرای گوشی دستی تأیید نشده‌اند. فاصله‌سنجی
چندهدف فعلی دوربین ثابت و زمین تخت کالیبره‌شده می‌خواهد؛ بدون کالیبراسیون فاصله null است.

## اجرای سریع

دستورها را از ریشه پروژه اجرا کنید. Python 3.11 یا 3.12 پیشنهاد می‌شود.

```sh
python -m pip install -r requirements.txt
python download_optional_model.py
python run_metric.py --source video.mp4 --model models/visdrone26m_1280.onnx --output observations.jsonl --show
```

`video.mp4` را با مسیر ویدئوی خود جایگزین کنید؛ برای دوربین محلی `--source 0`.
مدل دانلودی بزرگ و آموزش‌دیده روی تصاویر هوایی است؛ انتخاب نهایی گوشی یا
تضمین دقت در تصاویر زمینی نیست. وزن‌ها در Git نگهداری نمی‌شوند؛ دانلود این
مدل با SHA-256 کنترل می‌شود. فایل خروجی موجود بازنویسی نمی‌شود.

داشبورد محلی:

```sh
python run_laptop.py --model models/visdrone26m_1280.onnx --device cpu
```

نشانی: `http://127.0.0.1:5000`. برای فعال‌کردن فاصله چندهدف، راهنمای متریک را بخوانید.
نصب‌کننده‌های قدیمی لپ‌تاپ/Termux به مدل‌های فهرست‌شده در `models/SHA256SUMS`
نیاز دارند؛ clone به‌تنهایی آن مدل‌ها را ندارد.

## مستندات

- [فاصله‌سنجی، کالیبراسیون، Windows/Termux و ارزیابی](docs/METRIC_RUNTIME_FA.md)
- [تنظیمات پیشرفته لپ‌تاپ و آموزش](docs/LAPTOP_GUIDE_FA.md)
- [بازبینی و اصلاحات فاصله](docs/RANGE_DEBUG_REVIEW_FA.md)
- [آزمایش اجرای موتور متریک](docs/METRIC_TEST_REPORT.md)
- [ممیزی معماری](docs/RD_AUDIT_100M_FA.md)
- [پژوهش مدل‌ها](docs/LAPTOP_RESEARCH_FA.md) و [مقایسه تاریخی](docs/LAPTOP_EXPERIMENTS_FA.md)
- [منشأ مدل‌ها](models/PROVENANCE_V4.json) و [مجوز وزن‌ها](models/ULTRALYTICS_LICENSE.txt)

`ACCEPTANCE_STATUS.json` و ارزیاب لپ‌تاپ مربوط به معیار تاریخی ۱۵۰ مترند؛
آن‌ها را با پذیرش فاصله‌سنجی ۱۰۰ متر یکی ندانید.

## ساختار

| مسیر | کاربرد |
|---|---|
| `run_metric.py` / `run_laptop.py` / `main.py` | اجرای ویدئو، لپ‌تاپ و داشبورد |
| `metric_range.py` / `distance_estimator.py` | فاصله چندهدف زمین و روش عرض معلوم |
| `calibrate_*.py` / `camera_geometry.py` | کالیبراسیون و اصلاح لنز |
| `evaluate_*.py` / `benchmark_*.py` | ارزیابی و آزمایش قابل تکرار |
| `train_two_class.py` / `tune_laptop.py` | آموزش و انتخاب آستانه |
| `templates/` | رابط مرورگر |
| `tests/` | تست‌های نرم‌افزاری |
| `docs/` | مستندات و گزارش‌های فعلی |
| `validation/laptop/` | شواهد مقایسه مدل‌ها؛ داده آموزشی نیست |
| `models/` | منشأ، مجوز و وزن‌های دانلودی محلی |

## تست

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

موفقیت تست نرم‌افزاری اثبات دقت میدانی نیست. گزارش‌های نسخه‌های قدیمی از
شاخه جاری حذف شده‌اند و در تاریخچه Git قابل بازیابی‌اند.
