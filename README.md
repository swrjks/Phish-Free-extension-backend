# Phish-Free Extension


**Phish-Free** is a Chrome extension that detects potentially malicious URLs and phishing attempts. It pairs with a local Python backend for secure, privacy-focused URL analysis—all processing happens on *your* machine.

## 📂 Directory Structure

```
Phish-Free-extension-backend/
│
├─ backend/           # Python backend (app.py, requirements.txt)
├─ extension/         # Chrome extension (manifest.json, popup.html, etc.)
└─ README.md          # This file
```

- `backend/` – Python server for URL analysis
- `extension/` – Chrome extension files
- `README.md` – Setup & usage instructions

## 💾 Download

1. Go to the [latest release](https://github.com/swrjks/Phish-Free-extension-backend/releases/latest)
2. Download **Phish-Free-extension-backend.zip** from Assets
3. Unzip to any folder (keep the structure intact)

## 🚀 Load in Chrome

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the **`extension/`** folder (must contain `manifest.json`)

> ⚠️ **Important**: Choose only the `extension/` folder, *not* the root repo folder.

## 🖥️ Run Backend Server

The extension needs the local backend running:

1. Open terminal/command prompt
2. Navigate: `cd path/to/Phish-Free-extension-backend/backend`
3. (Optional) Create virtual env:
   ```
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```
4. Install deps: `pip install -r requirements.txt`
5. Start server: `python app.py`

Server runs at `http://localhost:5000` (keep terminal open).

## ✅ How to Use

1. Load extension in Chrome
2. Visit any website
3. Click the **Phish-Free icon** 
4. Get instant analysis: score, label, & risk reasons

## ⚠️ Important Notes

- Backend **must** be running for analysis
- ✅ **100% local** – no data leaves your machine
- Requires **Python 3.11**
- Troubleshooting? Reinstall deps: `pip install -r requirements.txt --force-reinstall`
