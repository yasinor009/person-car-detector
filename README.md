# تشخیص انسان و خودرو

برنامه محلی Python برای تشخیص فقط دو کلاس `person` و `car`، ردیابی اهداف و
تخمین تقریبی فاصله با دوربین کالیبره‌شده. مدل‌های موجود عمومی هستند؛ دقت
تشخیص یا فاصله تا ۱۰۰ متر روی دوربین گوشی شما هنوز با داده میدانی تأیید نشده است.

## اجرای Termux

نسخه Termux را از F-Droid نصب کنید، سپس از ریشه پروژه:

```bash
bash install_termux.sh
python main.py --browser-camera
```

در مرورگر همان گوشی باز کنید:

```text
http://127.0.0.1:5000
```

اجازه دوربین پشتی را بدهید. اگر پورت اشغال بود:

```bash
python main.py --browser-camera --port 5001
```

و آدرس `http://127.0.0.1:5001` را باز کنید.

## اجرای لپ‌تاپ

Windows:

```text
install_laptop.bat
start_laptop.bat
```

Linux:

```bash
bash install_laptop.sh
.venv/bin/python run_laptop.py --profile balanced
```

پروفایل `cpu` سبک‌تر است. پروفایل `long-range` از مدل عمومی بزرگ‌تر،
ورودی ۱۲۸۰ و برش تصویر استفاده می‌کند و روی CPU کندتر است:

```bash
python run_laptop.py --profile long-range
```

## کالیبراسیون و فاصله

`calibrate_lens.py` اعوجاج لنز را کالیبره می‌کند و
`calibrate_multiview.py` فاصله کانونی را از چند اندازه‌گیری مرجع برازش می‌دهد.
فاصله تک‌دوربینه به اندازه واقعی هدف، زاویه، کادر تشخیص و کالیبراسیون وابسته است.
در وضعیت نامعتبر برنامه باید فاصله را نامشخص نمایش دهد.

## آموزش و ارزیابی

- `train_two_class.py`: آموزش اختصاصی person/car
- `evaluate_detector.py` و `evaluate_laptop.py`: Precision، Recall و F1
- `tune_laptop.py`: تنظیم آستانه روی validation
- `tests/`: آزمون‌های نرم‌افزاری
- `models/PROVENANCE_V4.json`: منشأ، قرارداد و مجوز مدل‌ها

داده آموزش، validation و test باید بر اساس ویدئو/محل جدا شوند. خروجی confidence
مدل معادل دقت اندازه‌گیری‌شده نیست.

## تست

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python verify_models.py
```

## ساختار

| مسیر | کاربرد |
|---|---|
| `main.py`, `app.py` | داشبورد و دوربین مرورگر |
| `drone_detector.py` | موتور سبک تشخیص در Termux (نام فایل برای سازگاری حفظ شده) |
| `run_laptop.py`, `laptop_detector.py` | موتور لپ‌تاپ و long-range |
| `distance_estimator.py` | فاصله با هندسه دوربین |
| `temporal_tracker.py` | شناسه و ردیابی |
| `models/` | سه مدل لازم، هش و مجوز |
| `templates/` | رابط کاربری |
