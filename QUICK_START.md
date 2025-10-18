# UFC Web Interface - Quick Start Guide

## 🚀 3-Step Setup

### 1. Install Dependencies
```bash
poetry install
```

### 2. Verify Installation
```bash
python verify_web_interface.py
```

### 3. Run the Pipeline
```bash
python app.py
```

**That's it!** Your browser will open to http://localhost:5000 with your predictions dashboard.

---

## 📖 Common Commands

```bash
# Standard run (auto-opens web interface)
python app.py

# Custom port
python app.py --port 8080

# No web interface (automation mode)
python app.py --no-web

# Force rebuild datasets
python app.py --force-rebuild

# Force retrain model
python app.py --retrain-model

# All options
python app.py --force-rebuild --retrain-model --port 8080
```

---

## 🌐 Pages

- **Dashboard** → http://localhost:5000/
  - All predictions with confidence levels
  - Summary statistics
  - Click fights for details

- **Model Performance** → http://localhost:5000/model-performance
  - Accuracy metrics
  - Feature importance chart
  - Training statistics

- **Fight Details** → http://localhost:5000/fight/FIGHT_ID
  - Predicted winner
  - Fighter profiles
  - Physical stats comparison
  - Fighting style analysis

---

## 🛑 Stop Server

Press **CTRL+C** in the terminal

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: No module named 'flask'" | Run `poetry install` or `pip install flask plotly` |
| "Address already in use" | Use different port: `python app.py --port 8080` |
| "No predictions available" | Run pipeline first: `python app.py` (completes all 6 steps before launching web) |
| Web won't open | Manually visit http://localhost:5000 in browser |

---

## 📚 Full Documentation

- [WEB_INTERFACE_README.md](WEB_INTERFACE_README.md) - Complete feature guide
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Detailed setup instructions
- [WEB_INTERFACE_SUMMARY.md](WEB_INTERFACE_SUMMARY.md) - Implementation overview

---

**Enjoy your predictions dashboard! 🥊📊**
