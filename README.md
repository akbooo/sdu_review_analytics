# SDU Reviews — 2GIS Analytics Dashboard

A dashboard for analyzing student reviews of SDU University collected from 2GIS.  
Includes statistics, categorization, AI insights via Groq API, and a monthly review browser.

---

## 📁 File Structure

```
streamlit/
├── app_2gis_updated.py               # main application file
├── sdu_overall_march_classified.csv  # dataset with reviews and categories
├── sdu_mark.webp                     # logo / image (if used)
├── requirements.txt                  # dependencies
└── README.md                         # this file
```

> All files must be placed in the **same folder**:
> ```python
> CSV_PATH = "sdu_overall_march_classified.csv"
> ```

---

## ⚙️ Installation & Setup

### 1. Navigate to the project folder

```bash
cd /path/path/path/streamlit
```

### 2. Activate the virtual environment

```bash
source /path/path/path/streamlit/venv/bin/activate
```

### 3. Install dependencies (first time only)

```bash
pip install -r requirements.txt
```

After installation, NLTK will automatically download Russian stopwords on first run — internet connection required.  
To download manually in advance:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 4. Run the application

```bash
python -m streamlit run app_2gis_updated.py
```

The app will open at: **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | web interface framework |
| `pandas` | CSV loading and data processing |
| `numpy` | vector computations (embeddings) |
| `matplotlib` | all charts and graphs |
| `scikit-learn` | TF-IDF text vectorization |
| `nltk` | Russian stopwords for keyword extraction |
| `requests` | Groq API calls for AI insights |

---

## 🗂️ Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | general statistics, trends, ratings by category |
| 🎓 Categories | per-category analysis across all 23 categories + AI insight |
| 📅 Timeline | monthly sentiment chart, AI insight by period, review browser |
| 🔍 Browse Reviews | top months by reactions, monthly review browser |

---

## 🤖 AI Insights (Groq API)

The dashboard uses the **Groq API** with the `llama-3.1-8b-instant` model to generate insights.  
The API key is embedded in the code (`GROQ_API_KEY` at the top of the file). If the key stops working:

1. Register at [console.groq.com](https://console.groq.com)
2. Create a new API key
3. Replace the value at the top of `app_2gis_updated.py`:
```python
GROQ_API_KEY = "your_new_key"
```

---

## 📋 Dataset Format

The CSV file must contain the following columns:

| Column | Description |
|---|---|
| `text` | review text |
| `date` | date in the format `18 ноября 2025` |
| `rating` | rating from 1 to 5 |
| `has_official_reply` | whether there is an official reply (`True`/`False`) |
| `reactions_total` | number of reactions |
| `categories` | categories separated by `; ` (e.g. `4. Teaching Quality; 11. Campus Life`) |

---

## 🛠️ Troubleshooting

**Port already in use:**
```bash
python -m streamlit run app_2gis_updated.py --server.port 8502
```

**Virtual environment won't activate:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**CSV read error:**  
Make sure `CSV_PATH` in the code matches your actual filename.
