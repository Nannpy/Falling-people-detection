# SafetyFall-Alert
The final projects in CarpeDieam Camp 2

🌿 Hydroponics AI Monitoring Platform

แดชบอร์ดตรวจติดตามระบบไฮโดรโพนิกส์แบบ IoT ที่พร้อมใช้งานในระดับ production พัฒนาด้วย Dash และ Plotly ออกแบบในสไตล์ SaaS โทนมืดแบบ Silicon Valley ที่ทันสมัย






✨ Features
ฟีเจอร์	คำอธิบาย
Real-time KPIs	แสดงค่า Health score, จำนวน anomaly, ดัชนีความเสถียร, และค่าล่าสุด
Anomaly Detection	ตรวจจับค่าผิดปกติด้วย Rolling z-score พร้อมแสดงจุดบนกราฟ
Health Scoring	คำนวณคะแนนความเสถียรถ่วงน้ำหนัก (pH 40%, TDS 30%, Temp 30%)
AI Insights	สรุปสถานะระบบเป็นภาษาธรรมชาติ
Interactive Charts	กราฟเส้น 3 กราฟโทนมืด พร้อม overlay จุด anomaly
Date Filtering	เลือกช่วงเวลาเองได้ผ่าน DateRangePicker
Resampling	รวมข้อมูลแบบ Raw / 5 นาที / 15 นาที / 1 ชั่วโมง
Events Table	ตารางแสดงเหตุการณ์ anomaly แบบแบ่งหน้า (pagination)
🏗️ Architecture

แพลตฟอร์มออกแบบตามหลัก 3-layer separation of concerns อย่างชัดเจน:

┌──────────────────────────────────────────┐
│            Presentation Layer            │
│  app.py — Dash layout, callbacks, UI     │
├──────────────────────────────────────────┤
│           Intelligence Layer             │
│  intelligence.py — anomaly detection,    │
│  health score, AI insight generation     │
├──────────────────────────────────────────┤
│              Data Layer                  │
│  data_layer.py — CSV ingestion,          │
│  cleansing, resampling                   │
└──────────────────────────────────────────┘
Data Layer (data_layer.py)

clean_data(filepath) — อ่านไฟล์ CSV, แปลง timestamp, ลบแถวที่ไม่ถูกต้อง, บังคับคอลัมน์เซนเซอร์ให้เป็นตัวเลข, ทำ interpolation ตามเวลา, ลบค่า pH ที่อยู่นอกช่วง (< 3 หรือ > 10) และคืนค่า DataFrame ที่ index เป็น datetime

resample_data(df, frequency) — รวมข้อมูลตามความถี่ที่กำหนด (5min, 15min, 1H) ด้วยค่าเฉลี่ย หากส่ง "raw" จะคืนข้อมูลเดิมโดยไม่ปรับความถี่

Intelligence Layer (intelligence.py)

detect_anomalies(df) — ใช้ rolling window ขนาด 30 ค่า เพื่อคำนวณขอบเขต mean ± 3σ หากเกินขอบเขตจะถูกระบุเป็น anomaly ในคอลัมน์ boolean (pH_anomaly, TDS_anomaly, temp_anomaly)

health_score(df) — คำนวณคะแนนความเสถียร 0–100 จาก coefficient of variation ของแต่ละเซนเซอร์ โดยถ่วงน้ำหนัก pH 40%, TDS 30%, Temp 30% และคืนค่าทั้งคะแนนและสถานะ (Healthy / Warning / Critical)

generate_insight(df, health, anomaly_count) — สร้างข้อความสรุปสถานะระบบในรูปแบบภาษามนุษย์

Presentation Layer (app.py)

Layout พัฒนาแบบ Pure Dash ไม่มี logic วิเคราะห์ข้อมูลภายในไฟล์นี้

เรียกใช้ฟังก์ชันจาก data layer และ intelligence layer

ใช้ callback เดียวอัปเดตทุก component เพื่อให้ state สอดคล้องกัน

🎨 Design System
Token	Hex	การใช้งาน
Background	#0F172A	สีพื้นหลังหลัก
Card Surface	#1F2937	พื้นหลังการ์ด
Primary Green	#22C55E	เส้นกราฟ pH / ตัวบ่งชี้สถานะ
Accent Green	#16A34A	Hover / element เน้น
Cyan	#06B6D4	เส้นกราฟ TDS
Amber	#F59E0B	เส้นกราฟอุณหภูมิ
Red	#EF4444	จุด anomaly
Muted Text	#9CA3AF	ข้อความรอง / label
White	#F9FAFB	ข้อความหลัก

Typography: ใช้ฟอนต์ Inter จาก Google Fonts

🚀 Getting Started
Prerequisites

Python 3.9+

pip

Installation
# เข้าไปยังโฟลเดอร์โปรเจกต์
cd dash_project

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน dashboard
python app.py

แอปพลิเคชันจะเปิดที่ http://127.0.0.1:8050

Dataset

วางไฟล์ hydroponics_data.csv ไว้ที่ root ของโปรเจกต์ โดยต้องมีคอลัมน์ดังนี้:

คอลัมน์	ชนิดข้อมูล	คำอธิบาย
timestamp	datetime	เวลาที่บันทึกข้อมูล
pH	float	ค่าความเป็นกรด-ด่าง
TDS	float	Total Dissolved Solids (ppm)
water_temp	float	อุณหภูมิน้ำ (°C)

คอลัมน์อื่น ๆ จะถูกเก็บไว้แต่จะไม่ถูกนำมา plot บนกราฟ

📋 กระบวนการทำความสะอาดข้อมูล (Data Cleaning)

แปลง Timestamp — ใช้ pd.to_datetime(errors='coerce') แถวที่แปลงไม่ได้จะถูกลบ

ลบข้อมูลซ้ำ — ลบ timestamp ที่ซ้ำกัน (เก็บแถวแรก)

บังคับเป็นตัวเลข — คอลัมน์ pH, TDS, water_temp แปลงเป็น numeric หากไม่ใช่ตัวเลข → NaN

Interpolation ตามเวลา — เติมค่าที่หายไปโดยอิงจาก timestamp รอบข้าง

ลบ Outlier — ลบค่า pH ที่อยู่นอกช่วง 3–10

เรียงลำดับเวลา — เรียงข้อมูลตาม timestamp จากเก่าไปใหม่

📂 โครงสร้างโปรเจกต์
dash_project/
├── app.py              # Presentation layer (Dash)
├── data_layer.py       # Data ingestion & cleansing
├── intelligence.py     # Anomaly detection & health scoring
├── hydroponics_data.csv
├── requirements.txt
└── README.md
🛡️ License

MIT
