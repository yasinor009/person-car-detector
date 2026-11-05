# تشخیص انسان و خودرو و تخمین فاصله

پروژه Python برای تشخیص دو کلاس `person` و `car`، ردیابی مستقل اهداف و
فاصله‌سنجی هندسی کالیبره‌شده.

**وضعیت: پژوهشی.** دقت ۱۰۰ متر و اجرای گوشی دستی تأیید نشده‌اند. فاصله‌سنجی
چندهدف فعلی دوربین ثابت و زمین تخت کالیبره‌شده می‌خواهد؛ بدون کالیبراسیون فاصله null است.

## اجرای سریع

دستورها را از ریشه پروژه اجرا کنید. Python 3.11 یا 3.12 پیشنهاد می‌شود.

```sh
python -m pip install -r requirements.txt
python download_models.py --profile mobile
python run_metric.py --source video.mp4 --model models/person_car_yolo26n.onnx --output observations.jsonl --show
```

`video.mp4` را با مسیر ویدئوی خود جایگزین کنید؛ برای دوربین محلی `--source 0`.
مدل پیش‌فرض عمومی COCO است، برای تصاویر معمولی زمینی؛ مدل هوایی از مسیر
اجرا حذف شده است. دو وزن ONNX سبک/بزرگ‌تر همراه مخزن هستند و با SHA-256
کنترل می‌شوند. فایل خروجی موجود بازنویسی نمی‌شود.
در اجرای موبایل، وانت و کامیون نیز در گروه خروجی «خودرو» قرار می‌گیرند؛
برای محدودکردن به سواری از `--passenger-cars-only` استفاده کنید.

داشبورد محلی:

```sh
python main.py --browser-camera
```

نشانی: `http://127.0.0.1:5000`. برای فعال‌کردن فاصله چندهدف، راهنمای متریک را بخوانید.
برای اهداف کوچک و دور: `python main.py --browser-camera --profile long-range`.
این حالت مدل s، ورودی ۱۲۸۰ و برش تصویر را فعال می‌کند و روی گوشی کندتر است.
تشخیص ۱۰۰ متر هنوز با داده واقعی گوشی تأیید نشده است.

## مستندات

- [فاصله‌سنجی، کالیبراسیون، Windows/Termux و ارزیابی](docs/METRIC_RUNTIME_FA.md)
- [تنظیمات پیشرفته لپ‌تاپ و آموزش](docs/LAPTOP_GUIDE_FA.md)
- [بازبینی و اصلاحات فاصله](docs/RANGE_DEBUG_REVIEW_FA.md)
- [آزمایش اجرای موتور متریک](docs/METRIC_TEST_REPORT.md)
- [ممیزی معماری](docs/RD_AUDIT_100M_FA.md)
- [پژوهش مدل‌ها](docs/LAPTOP_RESEARCH_FA.md) و [مقایسه تاریخی](docs/LAPTOP_EXPERIMENTS_FA.md)
- [اصلاح تشخیص وانت و دوربین زمینی](docs/GROUND_DETECTION_FIX_FA.md)
- [منشأ مدل‌های فعلی](models/GROUND_MODELS.json) و [مجوز وزن‌ها](models/ULTRALYTICS_LICENSE.txt)

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
