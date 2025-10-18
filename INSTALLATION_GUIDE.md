# UFC Web Interface - Installation Guide

## 📦 Step 1: Install Dependencies

The web interface requires Flask and Plotly. Install them using Poetry:

```bash
# Navigate to project directory
cd c:\Users\jared\Documents\ufc4\ufcscraper

# Install all dependencies including new web interface packages
poetry install
```

**OR** if you're using pip directly:

```bash
pip install flask plotly
```

## ✅ Step 2: Verify Installation

Test that Flask is installed correctly:

```bash
python -c "import flask; import plotly; print('✓ All web dependencies installed successfully')"
```

You should see: `✓ All web dependencies installed successfully`

## 🚀 Step 3: Run the Pipeline

Now run your UFC automation pipeline as usual:

```bash
python app.py
```

The pipeline will:
1. ✅ Check data freshness
2. ✅ Scrape new data (if needed)
3. ✅ Move completed events
4. ✅ Build ML datasets
5. ✅ Validate data integrity
6. ✅ Train model and make predictions
7. 🌐 **Launch web interface** (NEW!)

## 🌐 Step 4: Access the Web Interface

After Step 6 completes, the web interface will automatically:
- Start a local web server on **http://localhost:5000**
- Open your default browser to the dashboard

You should see a beautiful dashboard with all your fight predictions! 🥊

## 🎯 What You'll See

### Dashboard Page
- Summary statistics (total fights, high confidence predictions, avg confidence)
- Table of all upcoming fights with predictions
- Color-coded confidence levels
- Click any fight to see detailed analysis

### Model Performance Page
- Model accuracy and AUC metrics
- Feature importance chart (top 20 features)
- Training/prediction sample counts

### Fight Detail Pages
- Predicted winner with confidence
- Win probability charts
- Fighter profiles side-by-side
- Physical stats comparison
- Fighting style radar charts
- Advanced ML features

## 🛠️ Configuration Options

### Change the Port
If port 5000 is already in use:
```bash
python app.py --port 8080
```

### Skip Web Interface
For automation/headless mode:
```bash
python app.py --no-web
```

### Force Rebuild/Retrain
```bash
python app.py --force-rebuild --retrain-model
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
**Solution**: Install Flask and Plotly
```bash
poetry install
# OR
pip install flask plotly
```

### "Address already in use"
**Solution**: Either:
1. Use a different port: `python app.py --port 8080`
2. Kill the process using port 5000

### "No predictions available"
**Solution**:
1. Make sure you have run the pipeline at least once
2. Check that `ufc_data/fight_predictions_summary.csv` exists
3. Try running with `--force-rebuild --retrain-model`

### Web page shows but no data
**Solution**:
1. Check that all CSV files exist in `ufc_data/`
2. Look at terminal output for any error messages
3. Refresh the page (F5)

### Charts not rendering
**Solution**:
1. Make sure you have internet connection (for CDN resources)
2. Check browser console (F12) for JavaScript errors
3. Try a different browser (Chrome, Firefox, Edge)

## 📁 Files Created

The installation created these new files:

```
ufcscraper/
├── web_app.py                     # ✅ Flask application
├── templates/                     # ✅ HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── model_performance.html
│   └── fight_detail.html
├── static/                        # ✅ Static assets
│   ├── css/style.css
│   └── js/charts.js
WEB_INTERFACE_README.md            # ✅ Documentation
INSTALLATION_GUIDE.md              # ✅ This file
```

**Modified files:**
- `pyproject.toml` - Added Flask and Plotly dependencies
- `app.py` - Added Step 7 for web interface launch

## ✨ Features at a Glance

✅ **No changes to ML pipeline** - Your existing predictions are untouched
✅ **Beautiful UFC-themed design** - Red, black, and gold colors
✅ **Responsive layout** - Works on desktop, tablet, mobile
✅ **Interactive charts** - Plotly-powered visualizations
✅ **Detailed fighter stats** - Physical attributes, career records, fighting styles
✅ **Advanced analytics** - 100+ ML features displayed
✅ **Easy to use** - Just run `python app.py`

## 🎉 You're Ready!

That's it! Run `python app.py` and enjoy your new web interface.

**Press CTRL+C** when you want to stop the web server.

---

Need help? Check [WEB_INTERFACE_README.md](WEB_INTERFACE_README.md) for full documentation.

Happy predicting! 🥊📊
