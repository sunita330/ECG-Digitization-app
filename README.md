# 🫀 CardioScan AI — ECG Digitization

Convert ECG images into clean digital signals using image processing + deep learning (U-Net).

---

## 🚀 Live Demo
https://ecg-digitization-app-yxn6.onrender.com

## unet_ecg_best.pth 
https://drive.google.com/file/d/1-r_pKLKCmhdzfSquA-ia82N39WyY3AA6/view?usp=drive_link
---

## ✨ Features

* 📤 Upload ECG images (PNG/JPG)
* ⚙️ 3 Modes: Classical | U-Net | Hybrid
* 📊 Signal extraction + waveform plot
* ❤️ Heart Rate estimation
* 📈 Step-by-step processing visualization

---

## 🛠 Tech Stack

* **Backend:** Flask, PyTorch, OpenCV
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Render

---

## 📁 Project Structure

```
app.py
index.html
script.js
style.css
requirements.txt
render.yaml
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/sunita330/ECG-Digitization-app
cd ECG-Digitization-app

pip install -r requirements.txt
python app.py
```

Open → http://127.0.0.1:5000

---

## 🌐 API

**POST /process**

* Input: ECG image
* Output: signal plot + metrics

---

## 🚀 Deployment (Render)

* Uses `gunicorn app:app`
* Auto-deploy via GitHub
* Model loads from local file or URL

---

## ⚠️ Common Issues

* ❌ "Cannot reach Flask server" → remove localhost from JS
* ❌ 502 error → increase gunicorn timeout
* ❌ Model not loading → check `.pth` path or download URL

---

## 📄 License

MIT License

---

💙 Built using Flask + PyTorch
