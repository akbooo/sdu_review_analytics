import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import os
from bs4 import BeautifulSoup
import re
from datetime import datetime
from pathlib import Path

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("Protected App")
    st.write("Please enter the password to continue.")

    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False

if not check_password():
    st.stop()
import base64
import io
import json
import requests
import pandas as pd
import streamlit as st

def github_headers():
    return {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "streamlit-sdu-review-app",
    }

def update_github_file(path: str, text_content: str, message: str, sha: str | None = None):
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(text_content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=github_headers(), json=payload, timeout=30)

    if not r.ok:
        raise Exception(f"GitHub API error {r.status_code}: {r.text}")

    return r.json()

def update_github_file(path: str, text_content: str, message: str, sha: str | None = None):
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(text_content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=github_headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def save_dataframe_to_github(df_to_save: pd.DataFrame, repo_csv_path: str):
    old_content, sha = get_github_file(repo_csv_path)

    csv_text = df_to_save.to_csv(index=False, encoding="utf-8")

    result = update_github_file(
        path=repo_csv_path,
        text_content=csv_text,
        message=f"Update reviews dataset from Streamlit ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        sha=sha,
    )
    return result
def get_github_file(path: str):
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, headers=github_headers(), params={"ref": branch}, timeout=30)

    if r.status_code == 404:
        return None, None

    if not r.ok:
        try:
            error_json = r.json()
        except Exception:
            error_json = r.text
        raise Exception(f"GitHub GET failed: {r.status_code} | {error_json}")

    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    sha = data["sha"]
    return content, sha

def update_data_last_updated_file():
    path = "data_last_updated.txt"
    old_content, sha = get_github_file(path)
    text = datetime.now().strftime("%d %B %Y, %H:%M")

    update_github_file(
        path=path,
        text_content=text,
        message="Update data timestamp",
        sha=sha,
    )
def get_repo_data_date():
    content, _ = get_github_file("data_last_updated.txt")
    if content:
        return content.strip()
    return "неизвестно"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ── download stopwords if needed ──────────────────────────────────────────────
try:
    stopwords.words("russian")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ── config ────────────────────────────────────────────────────────────────────
MODEL        = "llama-3.1-8b-instant"
CSV_PATH     = "sdu_categorized_final.csv"

st.set_page_config(
    page_title="SDU Reviews — 2GIS Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── SDU University Brand Colors ──────────────────────────────────────────────
   Navy Blue:  #1e2557  (primary brand / backgrounds)
   Orange:     #e8832a  (accent / CTA)
   Light Navy: #2a3470  (sidebar / cards)
   Off-white:  #f5f6fa  (page background)
   ──────────────────────────────────────────────────────────────────────────── */
html, body, .main, .block-container { background-color: #f5f6fa !important; }
h1,h2,h3,h4 { color:#1e2557 !important; font-family:'Inter',sans-serif !important; font-weight:700 !important; }
* { font-family:'Inter',sans-serif !important; color:#1e2557 !important; }
section[data-testid="stSidebar"] { background-color:#1e2557 !important; }
section[data-testid="stSidebar"] * { color:#ffffff !important; }
section[data-testid="stSidebar"] .stRadio label { color:#ffffff !important; }
section[data-testid="stSidebar"] hr { border-color:#2a3470 !important; }
.stMultiSelect [data-baseweb="tag"] { background-color:#e8832a !important; color:white !important; border-radius:6px !important; border:none !important; }
.stMultiSelect [data-baseweb="select"] * { color:#1e2557 !important; }
.stMultiSelect input { background-color:white !important; }
.metric-card { background-color:white !important; padding:18px !important; border-radius:12px !important;
  box-shadow:0 2px 10px rgba(30,37,87,0.12) !important; border-left:4px solid #e8832a !important; margin-bottom:15px !important; }
.stButton>button { background-color:#e8832a !important; color:white !important; border-radius:8px !important;
  padding:10px 20px !important; border:none !important; font-weight:600 !important; }
.stButton>button:hover { background-color:#cf6d1a !important; transition:0.2s; }

/* ── Quote buttons in problem zones: white background, red border ── */
.quote-btn div[data-testid="stButton"] > button {
  background-color: #ffffff !important;
  color: #c0392b !important;
  border: 1.5px solid #fca5a5 !important;
  font-weight: 400 !important;
  font-size: 0.88em !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 10px 14px !important;
  line-height: 1.5 !important;
}
.quote-btn div[data-testid="stButton"] > button:hover {
  background-color: #fff5f5 !important;
  border-color: #c0392b !important;
  color: #7f1d1d !important;
}
table { background-color:white !important; border-radius:10px !important; overflow:hidden !important; }
.stSelectbox [data-baseweb="select"] { background-color:white !important; }

/* insight boxes */
.insight-box { background:white; border-radius:16px; padding:24px 28px; margin:12px 0;
  box-shadow:0 4px 18px rgba(30,37,87,0.14); border-left:6px solid #e8832a; }
.insight-headline { font-size:1.25em; font-weight:800; color:#1e2557; margin-bottom:10px; }
.insight-section-title { font-size:0.85em; font-weight:700; color:#e8832a;
  text-transform:uppercase; letter-spacing:0.06em; margin:14px 0 6px; }
.insight-deep { background:#eef0f8; border-radius:10px; padding:14px 18px; margin-top:10px;
  font-size:0.97em; line-height:1.7; color:#1e2557; border-left:4px solid #e8832a; }
.tag { display:inline-block; background:#e8832a22; color:#1e2557; border-radius:20px;
  padding:3px 14px; font-size:0.8em; margin:3px 2px; font-weight:600; border:1px solid #e8832a55; }
.like-item { color:#1c9e7e; margin:3px 0; font-size:0.95em; }
.dislike-item { color:#c0392b; margin:3px 0; font-size:0.95em; }
.review-card { background:#fff; padding:14px 18px; margin:7px 0; border-radius:12px;
  box-shadow:0 2px 8px rgba(30,37,87,0.10); }
.review-meta { color:#888; font-size:0.8em; margin-top:6px; }

/* selectbox SDU border */
.stSelectbox > div > div {
  border: 2px solid #e8832a !important;
  border-radius: 8px !important;
  background: white !important;
}
.stSelectbox > div > div:focus-within {
  border-color: #1e2557 !important;
  box-shadow: 0 0 0 2px rgba(232,131,42,0.3) !important;
}
/* selectbox label styling */
.stSelectbox label {
  font-size: 1.05em !important;
  font-weight: 700 !important;
  color: #1e2557 !important;
}

/* radio selected dot → orange (all radio buttons app-wide) */
:root { --primary-color: #e8832a !important; }
.stApp { --primary-color: #e8832a !important; }

/* Target the outer ring div */
[data-baseweb="radio"] > label > div:first-child {
  border-color: #e8832a !important;
}
/* Target the inner filled dot div */
[data-baseweb="radio"] > label > div:first-child > div {
  background-color: #e8832a !important;
}

/* radio label background should stay transparent */
[data-testid="stSidebar"] .stRadio label {
  background: transparent !important;
}

/* remove selected option highlight background */
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
  background: transparent !important;
}
            
/* ── Hide Material Icon text in expanders (renders as "keyboard_arrow_right") ── */
[data-testid="stExpander"] details summary p { font-weight:600 !important; color:#1e2557 !important; }

[data-testid="collapsedControl"] { font-size:0 !important; }
[data-testid="collapsedControl"]::after {
  content:"▶";
  font-size:20px !important;
  color:#e8832a !important;
  display:block;
}
[data-testid="stSidebar"][aria-expanded="true"] + [data-testid="collapsedControl"]::after { content:"◀"; }
[data-testid="collapsedControl"] span { display:none !important; }
[data-testid="collapsedControl"] button { font-size:0 !important; color:transparent !important; }
[data-testid="collapsedControl"] button::after {
  content:"▶";
  font-size:20px !important;
  color:#e8832a !important;
}
.metric-card {
  background-color: white !important;
  padding: 18px !important;
  border-radius: 12px !important;
  box-shadow: 0 2px 10px rgba(30,37,87,0.12) !important;
  border-left: 4px solid #e8832a !important;
  margin-bottom: 15px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
}

.metric-card h4 {
  margin: 0 0 6px 0 !important;
  line-height: 1.2 !important;
  font-size: 1.4em !important;
}

.metric-card h2 {
  margin: 0 !important;
  line-height: 1 !important;
  white-space: nowrap !important;
}
.university-card * {
  color:white !important;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())

def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>""", unsafe_allow_html=True)

def strip_cat_num(name):
    """Remove leading number prefix like '22. ' or '1. ' from category names."""
    return re.sub(r"^\d+\.\s*", "", str(name)).strip()

def rating_to_sentiment(r):
    if r >= 4: return "positive"
    if r <= 2: return "negative"
    return "neutral"

MONTHS_RU = {
    "января":"01","февраля":"02","марта":"03","апреля":"04",
    "мая":"05","июня":"06","июля":"07","августа":"08",
    "сентября":"09","октября":"10","ноября":"11","декабря":"12",
}
MONTH_NAMES = {
    1:"Январь",2:"Февраль",3:"Март",4:"Апрель",
    5:"Май",6:"Июнь",7:"Июль",8:"Август",
    9:"Сентябрь",10:"Октябрь",11:"Ноябрь",12:"Декабрь",
}
SENT_LABELS = {"positive": "Позитивный", "negative": "Негативный", "neutral": "Нейтральный"}

_NUM_TO_RU_MONTH = {
    1:"января",2:"февраля",3:"марта",4:"апреля",5:"мая",6:"июня",
    7:"июля",8:"августа",9:"сентября",10:"октября",11:"ноября",12:"декабря",
}

def parse_date(x):
    """Parse date to ISO YYYY-MM-DD. Handles Russian text, ISO, and Timestamp strings."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # ISO or Timestamp format: '2024-01-15' or '2024-01-15 00:00:00'
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    # Russian format: '15 января 2024'
    parts = s.split()
    if len(parts) == 3:
        d, m, y = parts
        m = MONTHS_RU.get(m.lower())
        return f"{y}-{m}-{d.zfill(2)}" if m else None
    return None

def ts_to_ru_date(ts):
    """Convert Timestamp back to Russian string '15 января 2024' for CSV saving."""
    try:
        t = pd.to_datetime(ts)
        if pd.isna(t):
            return None
        return f"{t.day} {_NUM_TO_RU_MONTH[t.month]} {t.year}"
    except Exception:
        return None

russian_stopwords = stopwords.words("russian")

def top_keywords(texts, n=10):
    if len(texts) < 2:
        return []
    try:
        tf = TfidfVectorizer(stop_words=russian_stopwords, max_df=0.9, min_df=1)
        X  = tf.fit_transform(texts)
        terms = tf.get_feature_names_out()
        sums  = X.sum(axis=0).A1
        idx   = sums.argsort()[::-1][:n]
        return [terms[i] for i in idx]
    except:
        return []

def tfidf_embeddings(texts):
    tf = TfidfVectorizer(stop_words=russian_stopwords, max_df=0.95,
                         min_df=1, max_features=300, sublinear_tf=True)
    return tf.fit_transform(texts).toarray()

def call_groq(prompt, max_tokens=1800):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты аналитик данных. Отвечай ТОЛЬКО строгим JSON без markdown и backtick-ов. Все значения в JSON должны быть на русском языке."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    try:
        resp   = requests.post(url, headers=headers, json=payload, timeout=40)
        result = resp.json()
        if "error" in result:
            return {"error": result["error"]}
        raw = result["choices"][0]["message"]["content"]
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# ── PROMPTS ───────────────────────────────────────────────────────────────────
def format_category_prompt(cat, data):
    return f"""Ты эксперт-аналитик, изучающий отзывы студентов об университете СДУ (Казахстан).

КАТЕГОРИЯ: {cat}
ДОЛЯ ОТ ВСЕХ ОТЗЫВОВ: {data['percent']:.2f}%
КЛЮЧЕВЫЕ СЛОВА: {", ".join(data["keywords"])}

РЕПРЕЗЕНТАТИВНЫЕ ЦИТАТЫ:
{chr(10).join(['- ' + q for q in data["quotes"]])}

ОТЗЫВЫ (до 20 штук):
{chr(10).join(['- ' + t[:250] for t in data["texts"][:20]])}

Верни ТОЛЬКО этот JSON (без markdown, все поля на русском):
{{
  "label": "короткий заголовок 3-5 слов",
  "what_users_like": ["пункт 1", "пункт 2", "пункт 3"],
  "what_users_dislike": ["пункт 1", "пункт 2", "пункт 3"],
  "insight": "Подробный анализ на 5-8 предложений: что происходит в этой категории, какие паттерны и проблемы прослеживаются, как это влияет на опыт студентов, какие конкретные рекомендации можно дать руководству университета на основе этих отзывов."
}}"""

def format_month_prompt(period, texts, keywords, top_cats, sentiment_summary):
    return f"""Ты эксперт-аналитик. Анализируй отзывы студентов об университете СДУ (Казахстан) за период: {period}.

РАСПРЕДЕЛЕНИЕ СЕНТИМЕНТА: {sentiment_summary}
ТОП КАТЕГОРИЙ МЕСЯЦА: {", ".join(top_cats)}
КЛЮЧЕВЫЕ СЛОВА: {", ".join(keywords)}

ОТЗЫВЫ (до 25 штук):
{chr(10).join(['- ' + t[:200] for t in texts[:25]])}

Определи, что было актуально и обсуждалось в этот период.
Верни ТОЛЬКО этот JSON (без markdown, все поля на русском):
{{
  "headline": "краткое резюме месяца в 5-8 словах",
  "hot_topics": ["тема 1", "тема 2", "тема 3", "тема 4"],
  "positive_highlights": ["что хвалили 1", "что хвалили 2", "что хвалили 3"],
  "negative_highlights": ["на что жаловались 1", "на что жаловались 2", "на что жаловались 3"],
  "insight": "Подробный нарратив на 4-6 предложений: что делало этот месяц особенным, какие события или тенденции отразились в отзывах, как менялось настроение студентов, что можно порекомендовать."
}}"""

# ── HTML parser for 2GIS reviews ──────────────────────────────────────────────
_MONTHS_NUM_TO_RU = {
    1:"января",2:"февраля",3:"марта",4:"апреля",5:"мая",6:"июня",
    7:"июля",8:"августа",9:"сентября",10:"октября",11:"ноября",12:"декабря",
}
_MONTHS_RU_TO_NUM = {v: k for k, v in _MONTHS_NUM_TO_RU.items()}

def normalize_2gis_date(date_text):
    """Convert 2GIS date string to CSV format: 'D Month YYYY'."""
    if not date_text:
        return None
    s = date_text.strip().lower()
    s = re.sub(r"\s*г\.?$", "", s).strip()
    today = pd.Timestamp.now()
    if s == "сегодня":
        return f"{today.day} {_MONTHS_NUM_TO_RU[today.month]} {today.year}"
    if s == "вчера":
        yest = today - pd.Timedelta(days=1)
        return f"{yest.day} {_MONTHS_NUM_TO_RU[yest.month]} {yest.year}"
    parts = s.split()
    if len(parts) == 3:
        day, month_ru, year = parts
        if month_ru in _MONTHS_RU_TO_NUM and day.isdigit() and year.isdigit():
            return f"{int(day)} {month_ru} {year}"
    if len(parts) == 2:
        day, month_ru = parts
        if month_ru in _MONTHS_RU_TO_NUM and day.isdigit():
            return f"{int(day)} {month_ru} {today.year}"
    return date_text

def parse_html_2gis(html_content):
    """Parse reviews from 2GIS HTML"""
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = []
    containers = soup.find_all("div", class_="_1k5soqfl")
    
    for c in containers:
        # Date and edited flag
        date_tag = c.find("div", class_="_a5f6uz")
        date_text = date_tag.get_text(strip=True) if date_tag else None
        edited = False
        if date_text:
            edited = "отредакт" in date_text.lower()
            date_text = date_text.replace(", отредактирован", "").replace(", отредактирована", "")
        date_text = normalize_2gis_date(date_text)

        # Review text
        text_tag = c.find("div", class_="_49x36f")
        text = None
        if text_tag:
            a = text_tag.find("a")
            text = a.get_text(strip=True) if a else None
        
        # Rating
        rating_container = c.find("div", class_="_1fkin5c")
        rating = None
        if rating_container:
            yellow_stars = rating_container.find_all("svg")
            rating = len(yellow_stars)
        
        # Official reply
        reply_block = c.find("div", class_="_nqaxddm")
        has_reply = bool(reply_block)
        
        # Reactions
        reaction_blocks = c.find_all("div", class_="_e296pg")
        total_reactions = 0
        for rb in reaction_blocks:
            num_tag = rb.find("div", class_="_11fxohc")
            if num_tag:
                span = num_tag.find("span")
                if span:
                    txt = span.get_text(strip=True)
                    if txt.isdigit():
                        total_reactions += int(txt)
        
        if text and rating is not None:
            reviews.append({
                "text": text,
                "date": date_text,
                "rating": rating,
                "has_official_reply": has_reply,
                "reactions_total": total_reactions,
                "edited": edited
            })
    
    return pd.DataFrame(reviews)

# ── Groq categorization for new reviews ───────────────────────────────────────
CATEGORIES_LIST = [
    "1. Охрана и пропускной режим", "2. Еда и столовая",
    "3. Поступление и приёмная комиссия", "4. Преподаватели и качество образования",
    "5. Портал и IT-системы", "6. Стоимость и финансы",
    "7. Трудоустройство и карьера", "8. Религия", "9. Феминизм",
    "10. Транспорт и расположение", "11. Атмосфера и студенческая жизнь",
    "12. Администрация и студ. отдел", "13. Дискриминация и этика",
    "14. Инфраструктура и кампус", "15. Нерелевантный / пустой отзыв",
    "16. Лагерь", "17. Общежитие", "18. Дипломы и документы",
    "19. Отзывы выпускников", "20. Колл центр", "21. Общий негативный отзыв",
    "22. Общий нейтральный отзыв", "23. Общий позитивный отзыв",
    "Жалоба на удаление отзыва", "Упоминание имен учителей",
]
CATEGORIES_STR = "\n".join(f"- {c}" for c in CATEGORIES_LIST)

def categorize_reviews_groq(texts, batch_size=10):
    """Categorize a list of review texts via Groq. Returns list of category strings."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        numbered = "\n".join([f"{j+1}. {t[:300]}" for j, t in enumerate(batch)])
        prompt = f"""Ты аналитик отзывов университета СДУ (Казахстан).

Вот список допустимых категорий:
{CATEGORIES_STR}

Для каждого отзыва определи 1-3 наиболее подходящие категории ТОЛЬКО из списка выше.
Верни ТОЛЬКО JSON массив, без markdown, без пояснений.
Формат: [{{"idx": 1, "categories": "4. Преподаватели и качество образования; 11. Атмосфера и студенческая жизнь"}}, ...]

Отзывы:
{numbered}
"""
        resp = call_groq(prompt, max_tokens=1200)
        batch_map = {}
        if isinstance(resp, list):
            for item in resp:
                if isinstance(item, dict) and "idx" in item and "categories" in item:
                    batch_map[int(item["idx"])] = item["categories"]
        elif isinstance(resp, dict):
            for v in resp.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "idx" in item and "categories" in item:
                            batch_map[int(item["idx"])] = item["categories"]
                    break
        for j in range(len(batch)):
            results.append(batch_map.get(j+1, "15. Нерелевантный / пустой отзыв"))
    return results

def merge_and_deduplicate(existing_df, new_df):
    """Merge new reviews with existing ones — existing data is never touched,
    only new reviews not already present in existing are appended."""
    # Ensure all columns from existing_df are present in new_df
    for col in existing_df.columns:
        if col not in new_df.columns:
            if col in ("edited", "has_official_reply"):
                new_df[col] = False
            else:
                new_df[col] = None

    # Standardize data types for important columns
    for col in ("edited", "has_official_reply"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(bool)
        if col in existing_df.columns:
            existing_df[col] = existing_df[col].astype(bool)

    # Normalize text for comparison: lowercase, collapse all whitespace
    def norm(s):
        return re.sub(r"\s+", " ", str(s).lower().strip())

    existing_texts = set(existing_df["text"].apply(norm))
    truly_new = new_df[~new_df["text"].apply(norm).isin(existing_texts)].copy()

    # Restore existing_df dates to Russian string format for safe CSV round-trip
    existing_save = existing_df.copy()
    # Use date_str if available, otherwise convert Timestamp back to Russian string
    if "date_str" in existing_save.columns:
        existing_save["date"] = existing_save["date_str"]
    else:
        existing_save["date"] = existing_save["date"].apply(ts_to_ru_date)
    # Drop all derived/computed columns — they are rebuilt by load_data on next load
    drop_cols = [c for c in ("date_str", "year", "month", "year_month", "year_month_str",
                              "clean_text", "sentiment", "category", "orig_idx") if c in existing_save.columns]
    existing_save = existing_save.drop(columns=drop_cols)
    drop_cols_new = [c for c in drop_cols if c in truly_new.columns]
    truly_new = truly_new.drop(columns=drop_cols_new)

    # Append truly new rows to the restored existing data
    combined = pd.concat([existing_save, truly_new], ignore_index=True)
    combined = combined.reset_index(drop=True)
    return combined

# ── load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["text"])
    df["text"]            = df["text"].astype(str).apply(clean)
    df["clean_text"]      = df["text"].str.lower().str.strip()
    df["rating"]          = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)
    df["reactions_total"] = pd.to_numeric(df["reactions_total"], errors="coerce").fillna(0).astype(int)
    df["sentiment"]       = df["rating"].apply(rating_to_sentiment)
    
    # Handle categories column - may not exist if data was parsed from HTML
    if "categories" in df.columns:
        df["categories_raw"] = df["categories"].astype(str).str.strip()
    else:
        df["categories_raw"] = "Без категории"
    
    df["category"]        = df["categories_raw"].str.split(";").str[0].str.strip().apply(strip_cat_num)
    df["date_str"]        = df["date"].apply(parse_date)
    df["date"]            = pd.to_datetime(df["date_str"], errors="coerce")
    df["year"]            = df["date"].dt.year
    df["month"]           = df["date"].dt.month
    df["year_month"]      = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year_month_str"]  = df["date"].dt.to_period("M").astype(str)
    
    # Handle edited column - may not exist if data was loaded from older CSV
    if "edited" not in df.columns:
        df["edited"] = False
    
    df = df.reset_index(drop=True)
    df["orig_idx"] = df.index
    return df

@st.cache_data
def make_exploded(df):
    # prevent the same review being added twice for the same category
    rows = []
    seen = set()
    for idx, row in df.iterrows():
        cats = [strip_cat_num(c.strip()) for c in str(row["categories_raw"]).split(";") if c.strip()]
        # Remove duplicates from categories list to prevent same review appearing multiple times
        unique_cats = []
        for cat in cats:
            if cat not in unique_cats:
                unique_cats.append(cat)
        
        for cat in unique_cats:
            key = (row["orig_idx"], cat)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "orig_idx":        row["orig_idx"],
                "text":            row["text"],
                "clean_text":      row["clean_text"],
                "rating":          row["rating"],
                "sentiment":       row["sentiment"],
                "reactions_total": row["reactions_total"],
                "date":            row["date"],
                "year":            row["year"],
                "month":           row["month"],
                "year_month":      row["year_month"],
                "year_month_str":  row["year_month_str"],
                "category":        cat,
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_narxoz():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "narxoz.csv")
    dn = pd.read_csv(path, encoding="utf-8-sig")
    dn.columns = dn.columns.str.strip()
    dn = dn.dropna(subset=["text"])
    dn["text"]            = dn["text"].astype(str).apply(clean)
    dn["rating"]          = pd.to_numeric(dn["rating"], errors="coerce").fillna(3).astype(int)
    dn["reactions_total"] = pd.to_numeric(dn["reactions_total"], errors="coerce").fillna(0).astype(int)
    dn["sentiment"]       = dn["rating"].apply(rating_to_sentiment)
    dn["has_official_reply"] = dn["has_official_reply"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
    dn["date_str"]        = dn["date"].apply(parse_date)
    dn["date"]            = pd.to_datetime(dn["date_str"], errors="coerce")
    dn["year"]            = dn["date"].dt.year
    dn["month"]           = dn["date"].dt.month
    dn["year_month"]      = dn["date"].dt.to_period("M").dt.to_timestamp()
    return dn

@st.cache_data
def compute_tfidf_embeddings(texts):
    return tfidf_embeddings(texts)

df         = load_data()
df_exp     = make_exploded(df)
embeddings = compute_tfidf_embeddings(df["clean_text"].tolist())
all_categories = sorted(df_exp["category"].unique().tolist())

# ── review card renderer ──────────────────────────────────────────────────────
def render_review_cards(dataframe, max_n=20, key_prefix="rc"):
    sort_by = st.radio("Сортировать по:", ["Реакции (↓)", "Рейтинг (↓)", "Рейтинг (↑)"], horizontal=True, key=f"{key_prefix}_sort")
    if sort_by == "Реакции (↓)":
        dataframe = dataframe.sort_values("reactions_total", ascending=False)
    elif sort_by == "Рейтинг (↓)":
        dataframe = dataframe.sort_values("rating", ascending=False)
    else:
        dataframe = dataframe.sort_values("rating", ascending=True)

    total = len(dataframe)
    if total == 0:
        st.info("Отзывов нет.")
        return
    max_slider = min(50, total)
    default_n  = min(max_n, total)
    if max_slider > 1:
        show_n = st.slider("Показать отзывов:", 1, max_slider, default_n)
    else:
        show_n = total

    def _card_html(text, date_html, sent_color, cats_display, row):
        return f"""
        <div class="review-card" style="border-left:4px solid {sent_color};">
            <p style="margin:0;color:#1e2557;line-height:1.6;word-break:break-word;">{text}</p>
            <div style="display:flex;align-items:flex-end;margin-top:6px;flex-wrap:wrap;gap:8px;">
                <div>{date_html}</div>
                <p class="review-meta" style="margin:0;">
                    ⭐ {row.rating} &nbsp;|&nbsp; 👍 {row.reactions_total} реакций
                    &nbsp;|&nbsp; <span style="color:#e8832a;">{cats_display}</span>
                </p>
            </div>
        </div>
        """

    for idx, row in enumerate(dataframe.head(show_n).itertuples()):
        full_text = row.text
        is_long = len(full_text) > 350
        preview = full_text[:350] + "..." if is_long else full_text
        cats_display = " · ".join([strip_cat_num(c.strip()) for c in row.categories_raw.split(";") if c.strip()])
        sent_color = {"positive":"#39dfb6","negative":"#fe7070","neutral":"#ffbe51"}.get(row.sentiment, "#ccc")
        try:
            date_val = row.date
            date_display = pd.to_datetime(date_val).strftime("%d.%m.%Y") if pd.notna(date_val) else None
        except:
            date_display = None
        date_html = (
            f'<span style="display:inline-block;border:1.5px solid #e8832a;'
            f'border-radius:6px;padding:1px 8px;font-size:0.78em;'
            f'color:#e8832a;font-weight:600;background:#fff8f3;">{date_display}</span>'
            if date_display else ""
        )

        st.markdown(_card_html(preview, date_html, sent_color, cats_display, row), unsafe_allow_html=True)

        if is_long:
            with st.expander("Показать полный отзыв"):
                st.markdown(f"""
                <div style="color:#1e2557;line-height:1.8;font-size:0.96em;
                           word-break:break-word;white-space:pre-wrap;
                           font-family:'Inter',sans-serif;">{full_text}</div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;
                            padding-top:10px;border-top:1px solid #eee;">
                    <span style="background:#f0f2fa;padding:5px 12px;border-radius:20px;
                                 font-size:0.75em;color:#1e2557;font-weight:700;">
                        {row.rating} звёзд</span>
                    <span style="background:#f0f2fa;padding:5px 12px;border-radius:20px;
                                 font-size:0.75em;color:#1e2557;font-weight:700;">
                        {row.reactions_total} реакций</span>
                    {f'<span style="background:#fff8f3;padding:5px 12px;border-radius:20px;font-size:0.75em;color:#e8832a;font-weight:700;">{date_display}</span>' if date_display else ''}
                </div>
                """, unsafe_allow_html=True)

# ── patch sidebar collapse button text via JS ─────────────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
function patchSidebarButton() {
  const doc = window.parent.document;
  // Find all spans that contain material icon text
  doc.querySelectorAll('span, p').forEach(s => {
    const t = s.textContent.trim();
    if (t === 'keyboard_double_arrow_right' || t === 'keyboard_double_arrow_left') {
      s.textContent = t === 'keyboard_double_arrow_right' ? '▶' : '◀';
      s.style.fontFamily = 'Arial, sans-serif';
      s.style.fontSize = '16px';
      s.style.fontWeight = 'bold';
      s.style.color = '#e8832a';
    }
    if (t === 'keyboard_arrow_right') {
      s.textContent = '›';
      s.style.fontFamily = 'Arial, sans-serif';
      s.style.fontSize = '18px';
      s.style.color = '#e8832a';
    }
    if (t === 'keyboard_arrow_down') {
      s.textContent = '⌄';
      s.style.fontFamily = 'Arial, sans-serif';
      s.style.fontSize = '18px';
      s.style.color = '#e8832a';
    }
  });
}
patchSidebarButton();
const observer = new MutationObserver(patchSidebarButton);
observer.observe(window.parent.document.body, { childList: true, subtree: true, characterData: true });
</script>
""", height=0)

# ── sidebar logo + title ─────────────────────────────────────────────────────
import base64 as _b64, os as _os
_logo_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sdu_mark.webp")
if _os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = _b64.b64encode(_f.read()).decode()
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding: 18px 0 10px 0;">
      <img src="data:image/webp;base64,{_logo_b64}"
           style="width:170px; height:170px; object-fit:contain; border-radius:12px;" />
      <div style="margin-top:10px; font-size:1.05em; font-weight:800;
                  color:#ffffff; line-height:1.3;">
        Анализ Отзывов СДУ
      </div>
      <div style="font-size:0.75em; color:#e8832a; margin-top:2px;">
        2GIS · {len(df)} отзывов
      </div>
    </div>
    <hr style="border:none;border-top:1px solid #2a3470;margin:0 0 10px 0;">
    """, unsafe_allow_html=True)

# ── navigation ────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Навигация",
    [ "📊 Обзор", "🎓 Категории", "📅 Временная лента", "🏆 Сравнение"],
    label_visibility="collapsed"
)

# Scroll to top on every page change — counter forces fresh HTML each time
if st.session_state.get("current_page") != page:
    st.session_state["current_page"] = page
    st.session_state["scroll_count"] = st.session_state.get("scroll_count", 0) + 1

_scroll_n = st.session_state.get("scroll_count", 0)
_components.html(
    f"""
    <script>
    (function() {{
        var _t = {_scroll_n};
        function doScroll() {{
            var doc = window.parent.document;
            ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]',
             '[data-testid="stAppViewBlockContainer"]', '.main > div', '.main'
            ].forEach(function(sel) {{
                var el = doc.querySelector(sel);
                if (el) {{ el.scrollTop = 0; }}
            }});
            window.parent.scrollTo(0, 0);
            window.parent.document.documentElement.scrollTop = 0;
            window.parent.document.body.scrollTop = 0;
        }}
        doScroll();
        [100, 300, 600].forEach(function(ms) {{ setTimeout(doScroll, ms); }});
    }})();
    </script>
    """,
    height=0,
    scrolling=False,
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ОБЗОР
# ═════════════════════════════════════════════════════════════════════════════

if page == "📊 Обзор":
    st.title("📊 Анализ Отзывов СДУ")
    
    # ── Upload and update data section ─────────────────────────────────────────
    # ── Upload section — manual open/close via session_state ─────────────────
    if "show_upload" not in st.session_state:
        st.session_state["show_upload"] = False

    if st.button(
        "Обновить данные из HTML файла 2GIS" if not st.session_state["show_upload"] else "Скрыть загрузку",
        key="toggle_upload"
    ):
        st.session_state["show_upload"] = not st.session_state["show_upload"]
        st.rerun()

    if st.session_state["show_upload"]:
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:20px 24px;
                    margin-bottom:16px;box-shadow:0 2px 8px rgba(30,37,87,0.08);
                    border-left:4px solid #e8832a;">
        <b>Как это работает:</b><br>
        1. Откройте страницу СДУ на 2GIS<br>
        2. Прокрутите все отзывы<br>
        3. Нажмите Ctrl+S и сохраните страницу как HTML файл<br>
        4. Загрузите файл ниже<br><br>
        Приложение распарсит отзывы, нормализует даты, удалит дубликаты и присвоит категории через Groq AI.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Выберите HTML файл", type=["html"], key="html_uploader")

        if uploaded_file is not None:
            try:
                html_content = uploaded_file.read().decode("utf-8")
                new_reviews_df = parse_html_2gis(html_content)

                if len(new_reviews_df) > 0:
                    st.success(f"✅ Распарсено {len(new_reviews_df)} отзывов из HTML")

                    with st.expander("Примеры новых отзывов", expanded=False):
                        preview_cols = ["text", "date", "rating", "reactions_total", "edited"]
                        st.dataframe(new_reviews_df[preview_cols].head(5), use_container_width=True)

                    def norm(s):
                        return re.sub(r"\s+", " ", str(s).lower().strip())

                    existing_texts = set(df["text"].apply(norm))
                    duplicates_removed = new_reviews_df["text"].apply(norm).isin(existing_texts).sum()
                    truly_new_count = int(len(new_reviews_df) - duplicates_removed)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Старых отзывов", len(df))
                    col2.metric("Новых отзывов", len(new_reviews_df))
                    col3.metric("Дубликатов (уже есть)", int(duplicates_removed))
                    col4.metric("Добавится новых", truly_new_count)

                    if truly_new_count == 0:
                        st.info("ℹ️ Все отзывы из файла уже есть в базе. Ничего не добавится.")
                    else:
                        st.markdown("---")
                        st.markdown("**Категоризация новых отзывов через Groq AI**")
                        st.caption(f"Нужно категоризировать {truly_new_count} новых отзывов батчами по 10.")

                        if st.button("Категоризировать и сохранить", key="categorize_save", use_container_width=True):
                            truly_new_df = new_reviews_df[
                                ~new_reviews_df["text"].apply(norm).isin(existing_texts)
                            ].copy().reset_index(drop=True)

                            progress_bar = st.progress(0, text="Категоризирую отзывы...")
                            total_batches = (len(truly_new_df) + 9) // 10
                            all_categories = []

                            for batch_i in range(total_batches):
                                batch_texts = truly_new_df["text"].tolist()[batch_i*10:(batch_i+1)*10]
                                batch_cats = categorize_reviews_groq(batch_texts, batch_size=10)
                                all_categories.extend(batch_cats)

                                progress_bar.progress(
                                    (batch_i + 1) / total_batches,
                                    text=f"Батч {batch_i+1} из {total_batches}..."
                                )

                            progress_bar.empty()

                            truly_new_df["categories"] = all_categories
                            truly_new_df["categories_raw"] = all_categories

                            merged_df = merge_and_deduplicate(df, truly_new_df)

                            with st.spinner("Сохраняю данные в GitHub..."):
                                save_dataframe_to_github(merged_df, CSV_PATH)
                                update_data_last_updated_file()

                            st.cache_data.clear()
                            st.session_state["show_upload"] = False
                            st.success(
                                f"✅ Добавлено {truly_new_count} новых отзывов с категориями и сохранено в GitHub!"
                            )
                            st.rerun()
                else:
                    st.warning("⚠️ Не удалось распарсить отзывы из HTML. Проверьте формат файла.")

            except Exception as e:
                st.error(f"❌ Ошибка при обработке файла: {str(e)}")    

    MONTH_NAMES = {
        1: "января", 2: "февраля", 3: "марта", 4: "апреля",
        5: "мая", 6: "июня", 7: "июля", 8: "августа",
        9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
    }

    repo_date = get_repo_data_date()

    st.markdown(
        f'''
        <p style="color:#888;font-size:0.85em;margin-top:-10px;margin-bottom:18px;">
            🕐 Данные последний раз обновлены:
            <strong style="color:#e8832a;">{repo_date}</strong>
        </p>
        ''',
        unsafe_allow_html=True
    )
    # ── Executive Summary ─────────────────────────────────────────────────────
    avg_r     = df["rating"].mean()
    pct_pos   = (df.sentiment == "positive").mean() * 100
    pct_neg   = (df.sentiment == "negative").mean() * 100
    pct_reply = (df.has_official_reply == True).mean() * 100
    pct_edited = (df["edited"] == True).mean() * 100 if "edited" in df.columns else 0

    df_sorted = df.dropna(subset=["year_month"]).sort_values("year_month")
    months_all = sorted(df_sorted["year_month"].unique())
    half = max(1, len(months_all) // 2)
    avg_recent  = df_sorted[df_sorted["year_month"].isin(months_all[-half:])]["rating"].mean()
    avg_earlier = df_sorted[df_sorted["year_month"].isin(months_all[:half])]["rating"].mean()
    trend_delta = avg_recent - avg_earlier
    trend_arrow = "" if trend_delta > 0.1 else (""if trend_delta < -0.1 else "➡️")
    trend_label = "рост" if trend_delta > 0.1 else ("снижение" if trend_delta < -0.1 else "стабильно")

    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:24px 28px;
                box-shadow:0 2px 12px rgba(30,37,87,0.10);
                border-left:6px solid #e8832a;margin-bottom:20px;">
      <div style="font-size:1.1em;font-weight:800;color:#1e2557;margin-bottom:18px;">
        🎯 Ключевые показатели
      </div>
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:16px;">
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#e8832a;line-height:1.2;">{avg_r:.2f}</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Средний рейтинг</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#39dfb6;line-height:1.2;">{pct_pos:.0f}%</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Позитивных</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#fe7070;line-height:1.2;">{pct_neg:.0f}%</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Негативных</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#1e2557;line-height:1.2;">{len(df)}</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Всего отзывов</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#1e2557;line-height:1.2;">{abs(pct_reply):.0f}% {trend_arrow}</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Отвечено</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <div style="font-size:2em;font-weight:900;color:#9b59b6;line-height:1.2;">{pct_edited:.0f}%</div>
          <div style="font-size:0.78em;color:#888;margin-top:4px;text-align:center;">Отредактировано</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()


    # ── rating distribution + yearly sentiment side by side ───────────────────
    col_r, col_y = st.columns(2)

    with col_r:
        st.subheader("Распределение рейтингов")
        fig, ax = plt.subplots(figsize=(6, 3))
        counts = [len(df[df["rating"]==r]) for r in [1,2,3,4,5]]
        bars = ax.bar([1,2,3,4,5], counts, color="#e8832a", edgecolor="none", width=0.6)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(cnt), ha='center', va='bottom', fontsize=9, color="#1e2557")
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlabel("Рейтинг"); ax.set_ylabel("Кол-во")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col_y:
        st.subheader("Доля сентимента по годам")
        yearly = (df.groupby("year")["sentiment"]
                    .value_counts(normalize=True).unstack().fillna(0)
                    .rename(columns=SENT_LABELS))
        fig, ax = plt.subplots(figsize=(6, 3))
        yearly.plot(kind="bar", stacked=True, ax=ax,
                    color={"Позитивный":"#39dfb6","Негативный":"#fe7070","Нейтральный":"#ffda3b"})
        ax.set_xlabel("Год"); ax.set_ylabel("Доля")
        ax.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ── timeline (all time) ───────────────────────────────────────────────────
    st.subheader("Динамика отзывов по времени")
    timeline = (df.groupby(["year_month","sentiment"]).size().unstack(fill_value=0)
                  .rename(columns=SENT_LABELS))
    fig, ax = plt.subplots(figsize=(14, 4))
    for sent, color in [("Позитивный","#39dfb6"),("Негативный","#fe7070"),("Нейтральный","#ffda3b")]:
        if sent in timeline:
            ax.plot(timeline.index, timeline[sent], label=sent, linewidth=2, marker="o",
                    color=color, markersize=4)
    ax.set_xlabel("Время"); ax.set_ylabel("Кол-во")
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=0); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── avg rating per category ───────────────────────────────────────────────
    st.subheader("Средний рейтинг по категориям")
    avg_rating_cat = df_exp.groupby("category")["rating"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = ["#fe7070" if v < 3 else "#39dfb6" if v >= 4 else "#ffda3b"
                  for v in avg_rating_cat.values]
    avg_rating_cat.plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_xlabel("Средний рейтинг")
    ax.axvline(x=3, color="#e8832a", linestyle="--", alpha=0.4, label="Нейтральная отметка (3)")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(); plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Самые негативные категории ────────────────────────────────────────────
    st.subheader("🔴 Самые негативные категории")
    st.caption("Категории с наибольшей долей негативных отзывов (% от всех отзывов в категории)")

    skip_cats_ov = {"Нерелевантный / пустой отзыв", "Общий негативный отзыв"}
    neg_pct_data = []
    for cat, grp in df_exp.groupby("category"):
        if cat in skip_cats_ov:
            continue
        total_cat = len(grp)
        if total_cat < 5:
            continue
        neg_cnt = (grp["sentiment"] == "negative").sum()
        neg_pct = neg_cnt / total_cat * 100
        neg_pct_data.append({"category": cat, "neg_pct": neg_pct, "neg_cnt": int(neg_cnt), "total": total_cat})

    neg_pct_df = pd.DataFrame(neg_pct_data).sort_values("neg_pct", ascending=False).head(12)

    if len(neg_pct_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(neg_pct_df) * 0.55)))
        y_pos = range(len(neg_pct_df))
        bars = ax.barh(
            list(y_pos),
            neg_pct_df["neg_pct"].values,
            color="#fe7070",
            edgecolor="#c0392b",
            linewidth=0.7,
            height=0.6,
        )
        # Add percentage labels inside/outside bars
        for bar, pct, cnt, tot in zip(bars,
                                       neg_pct_df["neg_pct"].values,
                                       neg_pct_df["neg_cnt"].values,
                                       neg_pct_df["total"].values):
            label_x = bar.get_width() + 0.8
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%  ({cnt}/{tot})",
                    va="center", ha="left", fontsize=8.5, color="#1e2557", fontweight="600")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(neg_pct_df["category"].tolist(), fontsize=9)
        ax.set_xlabel("Доля негативных отзывов (%)")
        ax.set_xlim(0, min(100, neg_pct_df["neg_pct"].max() + 20))
        ax.axvline(x=50, color="#e8832a", linestyle="--", alpha=0.4, linewidth=1, label="50%")
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Недостаточно данных для построения графика.")

    st.divider()

    # ── Поиск отзывов по словам ───────────────────────────────────────────────
    st.subheader("🔍 Поиск по отзывам")
    st.caption("Введите слово или фразу для поиска по всем отзывам")

    search_query = st.text_input(
        "Поисковый запрос",
        placeholder="Например: преподаватель, столовая, wifi, общежитие...",
        key="ob_search_query",
        label_visibility="collapsed",
    )

    if search_query.strip():
        q = search_query.strip().lower()

        def highlight(text, query):
            import re as _re
            escaped = _re.escape(query)
            return _re.sub(
                f"({escaped})",
                r'<mark style="background:#fff3cd;color:#1e2557;border-radius:3px;padding:0 2px;">\1</mark>',
                text,
                flags=_re.IGNORECASE,
            )

        pattern = rf"\b{re.escape(q)}\b"

        search_results = df[
            df["text"].str.contains(pattern, case=False, na=False, regex=True)
        ].copy()
        st.markdown(
            f'<div style="font-weight:700;font-size:1em;color:#1e2557;margin-bottom:12px;">'
            f'Найдено: <span style="color:#e8832a;">{len(search_results)}</span> отзывов</div>',
            unsafe_allow_html=True,
        )

        if len(search_results) == 0:
            st.info("По вашему запросу отзывов не найдено.")
        else:
            show_n = min(30, len(search_results))
            for _, row in search_results.head(show_n).iterrows():
                text_hl = highlight(row["text"], q)
                sent_color = {
                    "positive": "#39dfb6",
                    "negative": "#fe7070",
                    "neutral":  "#ffbe51",
                }.get(row.get("sentiment", "neutral"), "#ccc")
                try:
                    date_val = row["date"]
                    date_display = pd.to_datetime(date_val).strftime("%d.%m.%Y") if pd.notna(date_val) else None
                except Exception:
                    date_display = None
                date_badge = (
                    f'<span style="border:1.5px solid #e8832a;border-radius:6px;'
                    f'padding:1px 8px;font-size:0.75em;color:#e8832a;font-weight:600;'
                    f'background:#fff8f3;">📅 {date_display}</span>'
                    if date_display else ""
                )
                cats_display = " · ".join([
                    strip_cat_num(c.strip())
                    for c in str(row.get("categories_raw", "")).split(";") if c.strip()
                ])
                st.markdown(
                    f'<div style="background:white;border-left:4px solid {sent_color};'
                    f'border-radius:10px;padding:12px 16px;margin-bottom:8px;'
                    f'box-shadow:0 1px 6px rgba(30,37,87,0.08);">'
                    f'<p style="margin:0;color:#1e2557;line-height:1.6;font-size:0.93em;word-break:break-word;">{text_hl}</p>'
                    f'<div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">'
                    f'{date_badge}'
                    f'<span style="font-size:0.78em;color:#888;">⭐ {int(row["rating"])} &nbsp;|&nbsp; '
                    f'👍 {int(row["reactions_total"])} реакций</span>'
                    + (f'<span style="font-size:0.78em;color:#e8832a;">{cats_display}</span>' if cats_display else '')
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )
            if len(search_results) > show_n:
                st.caption(f"Показано {show_n} из {len(search_results)} отзывов.")
    else:
        st.markdown(
            '<div style="background:#f0f2fa;border-radius:10px;padding:14px 18px;'
            'color:#888;font-size:0.92em;">Введите запрос выше, чтобы найти отзывы.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Сравнение годов ───────────────────────────────────────────────────────
    st.subheader("📅 Сравнение годов")
    years_avail_ob = sorted(df["year"].dropna().unique().astype(int), reverse=True)
    if len(years_avail_ob) >= 2:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            yr_new_ob = st.selectbox("📅 Год А", years_avail_ob, index=0, key="ob_yr_new")
        with col_sel2:
            yr_old_ob = st.selectbox("📅 Год Б", [y for y in years_avail_ob if y != yr_new_ob], index=0, key="ob_yr_old")

        df_new_ob = df[df["year"] == yr_new_ob]
        df_old_ob = df[df["year"] == yr_old_ob]

        def ob_yr_stats(d):
            return {
                "Кол-во отзывов":  len(d),
                "Средний рейтинг": round(d["rating"].mean(), 2),
                "Позитивных %":    round((d.sentiment=="positive").mean()*100, 1),
                "Негативных %":    round((d.sentiment=="negative").mean()*100, 1),
                "% с ответом":     round((d.has_official_reply==True).mean()*100, 1),
            }

        s_new_ob = ob_yr_stats(df_new_ob)
        s_old_ob = ob_yr_stats(df_old_ob)

        comp_cols_ob = st.columns(len(s_new_ob))
        for col, (metric, val_new) in zip(comp_cols_ob, s_new_ob.items()):
            val_old = s_old_ob[metric]
            if isinstance(val_new, float):
                delta = val_new - val_old
                delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
                delta_color = "#39dfb6" if delta > 0 else ("#fe7070" if delta < 0 else "#888")
                if "Негативных" in metric:
                    delta_color = "#fe7070" if delta > 0 else "#39dfb6"
            else:
                delta = val_new - val_old
                delta_str = f"+{delta}" if delta > 0 else str(delta)
                delta_color = "#39dfb6" if delta > 0 else ("#fe7070" if delta < 0 else "#888")
            col.markdown(f"""
            <div style="background:white;border-radius:12px;padding:14px 16px;
                        text-align:center;box-shadow:0 1px 8px rgba(30,37,87,0.08);
                        border-top:3px solid #e8832a;">
              <div style="font-size:0.75em;color:#888;margin-bottom:4px;">{metric}</div>
              <div style="font-size:1.4em;font-weight:900;color:#1e2557;">{val_new}
                <span style="font-size:0.5em;color:#aaa;font-weight:400;">({yr_new_ob})</span>
              </div>
              <div style="font-size:0.75em;color:#888;">{yr_old_ob}: {val_old}</div>
              <div style="font-size:0.85em;font-weight:700;color:{delta_color};">{delta_str}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Недостаточно данных для сравнения — нужно минимум 2 года.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: КАТЕГОРИИ
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎓 Категории":
    st.title("🎓 Аналитика по категориям")
    st.caption("Детальный анализ отзывов, сгруппированных по отдельным темам или категориям")

    # ── 1. Распределение по всем категориям (pie) — вверху страницы ──────────
    st.subheader("Распределение по категориям")
    cat_counts = df_exp["category"].value_counts()
    top12 = cat_counts.head(12).copy()
    other = cat_counts.iloc[12:].sum()
    if other > 0:
        top12["Другие"] = other
    fig, ax = plt.subplots(figsize=(9, 5))
    wedges, _, autotexts = ax.pie(
        top12, labels=None, autopct="%1.0f%%",
        startangle=140, pctdistance=0.80,
        colors=plt.cm.tab20.colors[:len(top12)]
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, top12.index, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── 2. Выбор категории ────────────────────────────────────────────────────
    st.subheader("📂 Выберите категорию")
    selected_cat = st.selectbox("", all_categories, label_visibility="collapsed")

    sub_exp      = df_exp[df_exp["category"] == selected_cat]
    orig_idxs    = sub_exp["orig_idx"].unique().tolist()
    subset_texts = sub_exp["text"].tolist()
    percent      = len(orig_idxs) / len(df) * 100
    keywords     = top_keywords(subset_texts)

    # representative quotes via embeddings
    sub_emb  = embeddings[orig_idxs]
    centroid = sub_emb.mean(axis=0)
    norms    = np.linalg.norm(sub_emb, axis=1) * np.linalg.norm(centroid)
    norms[norms == 0] = 1e-9
    sim      = (sub_emb @ centroid) / norms
    best_k   = min(3, len(orig_idxs))
    best_pos = np.argsort(sim)[::-1][:best_k]
    quotes   = [df.loc[orig_idxs[i], "text"] for i in best_pos]

    # ── metrics ───────────────────────────────────────────────────────────────
    avg_cat_rating = df.loc[orig_idxs, "rating"].mean() if orig_idxs else 0

    st.markdown(f"""
    <div style="
        background:white;
        border-radius:14px;
        padding:24px 28px;
        box-shadow:0 2px 12px rgba(30,37,87,0.10);
        border-left:6px solid #e8832a;
        margin-bottom:20px;
    "><div style="
            font-size:1.1em;
            font-weight:800;
            color:#1e2557;
            margin-bottom:18px;
        ">
            📊 Статистика категории
        </div><div style="
            display:flex;
            justify-content:space-between;
            gap:16px;
            text-align:center;
        "><div style="flex:1;">
                <div style="font-size:2em;font-weight:900;color:#1e2557;">{len(orig_idxs)}</div>
                <div style="font-size:0.82em;color:#888;">Отзывов</div>
            </div><div style="flex:1;">
                <div style="font-size:2em;font-weight:900;color:#e8832a;">{percent:.1f}%</div>
                <div style="font-size:0.82em;color:#888;">Доля</div>
            </div><div style="flex:1;">
                <div style="font-size:2em;font-weight:900;color:#39dfb6;">{avg_cat_rating:.1f} ★</div>
                <div style="font-size:0.82em;color:#888;">Средний рейтинг</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    # ── Сентимент (слева) + Рейтинги (справа) ───────────────────────────────
    color_map_ru = {"Позитивный":"#39dfb6","Негативный":"#fe7070","Нейтральный":"#ffda3b"}
    col_sent, col_hist = st.columns(2)
    with col_sent:
        st.subheader("Сентимент категории")
        sent_counts = sub_exp["sentiment"].value_counts().rename(index=SENT_LABELS)
        pie_colors  = [color_map_ru.get(s, "#e8832a") for s in sent_counts.index]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(sent_counts, labels=sent_counts.index, colors=pie_colors,
               autopct="%1.0f%%", startangle=90)
        st.pyplot(fig)

    with col_hist:
        st.subheader("Распределение рейтингов")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sub_exp["rating"], bins=[0.5,1.5,2.5,3.5,4.5,5.5],
                edgecolor="none", color="#e8832a")
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlabel("Рейтинг"); ax.set_ylabel("Кол-во")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ── AI Insight ────────────────────────────────────────────────────────────
    st.subheader("🧠 AI-инсайт по категории")
    cat_key = f"cat_insight_{selected_cat}"
    if st.button("✨ Сгенерировать инсайт", key="btn_cat_insight"):
        data = {"percent": percent, "keywords": keywords, "quotes": quotes, "texts": subset_texts}
        with st.spinner("Анализирую отзывы..."):
            analysis = call_groq(format_category_prompt(selected_cat, data), max_tokens=1800)
        if "error" in analysis:
            st.error(f"Ошибка: {analysis['error']}")
        else:
            st.session_state[cat_key] = analysis
    if cat_key in st.session_state:
        analysis      = st.session_state[cat_key]
        likes         = analysis.get("what_users_like", [])
        dislikes      = analysis.get("what_users_dislike", [])
        insight       = analysis.get("insight", "")
        label         = analysis.get("label", "")
        likes_html    = "".join([f'<p class="like-item">✅ {x}</p>' for x in likes])
        dislikes_html = "".join([f'<p class="dislike-item">❌ {x}</p>' for x in dislikes])
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-headline">🟦 {label}</div>
          <div style="color:#888;font-size:0.85em;margin-bottom:12px;">
            Доля отзывов в этой категории: <strong>{percent:.1f}%</strong>
          </div>
          <div class="insight-section-title">💙 Что студентам нравится</div>
          {likes_html}
          <div class="insight-section-title">💔 На что студенты жалуются</div>
          {dislikes_html}
          <div class="insight-section-title">🧠 Глубокий анализ</div>
          <div class="insight-deep">{insight}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── All reviews as cards at the bottom ────────────────────────────────────
    st.subheader("📋 Все отзывы категории")
    cat_reviews_df = df.loc[orig_idxs].reset_index(drop=True)
    render_review_cards(cat_reviews_df, key_prefix="cat")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ВРЕМЕННАЯ ЛЕНТА
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📅 Временная лента":
    st.title("📅 AI-таймлайн отзывов")
    st.caption("Анализ того, что было актуально в каждый период")

    # ── Сентимент по месяцам (2023–2026) — самый верх ────────────────────────
    st.subheader("Сентимент по месяцам")
    period_all = df[(df["year_month"] >= "2023-01-01") & (df["year_month"] <= "2026-12-31")]
    monthly_sent = (period_all.groupby(["year_month","sentiment"]).size()
                              .unstack(fill_value=0).rename(columns=SENT_LABELS))
    monthly_sent.index = monthly_sent.index.strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(16, 5))
    monthly_sent.plot(kind="bar", stacked=True, ax=ax,
                      color={"Позитивный":"#39dfb6","Негативный":"#fe7070","Нейтральный":"#ffda3b"})
    ax.set_xlabel("Месяц"); ax.set_ylabel("Кол-во отзывов")
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=45, fontsize=8); plt.tight_layout()
    st.pyplot(fig)


    # ── Топ 5 месяцев по реакциям ─────────────────────────────────────────────
    st.subheader("Топ 5 месяцев по реакциям")
    monthly_likes = (
        df.groupby("year_month_str")["reactions_total"]
        .sum().reset_index()
        .sort_values("reactions_total", ascending=False)
        .head(5)
    )
    monthly_likes = monthly_likes.merge(
        df.groupby("year_month_str").size().reset_index(name="count"),
        on="year_month_str", how="left"
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(monthly_likes["year_month_str"], monthly_likes["reactions_total"],
                   color="#e8832a", edgecolor="none", height=0.55)
    for bar, cnt in zip(bars, monthly_likes["count"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width())} реакций · {cnt} отзывов',
                va='center', ha='left', fontsize=9, color="#1e2557")
    ax.set_xlabel("Сумма реакций")
    ax.invert_yaxis()
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    df_dated = df.dropna(subset=["year_month"])

    st.markdown('<div class="period-card">', unsafe_allow_html=True)
    st.markdown('<div class="period-title">🗓️ Выберите период</div>', unsafe_allow_html=True)

    col_y, col_m = st.columns(2)

    with col_y:
        years = sorted(df_dated["year"].dropna().unique().astype(int), reverse=True)
        sel_year = st.selectbox("📅 Год", years, key="tl_year")

    with col_m:
        months_for_year = sorted(
            df_dated[df_dated["year"] == sel_year]["month"].dropna().unique().astype(int)
        )
        sel_month = st.selectbox(
            "🗓 Месяц",
            months_for_year,
            key="tl_month",
            format_func=lambda m: MONTH_NAMES.get(m, str(m))
        )

    st.markdown('</div>', unsafe_allow_html=True)

    sub_month = df[(df["year"] == sel_year) & (df["month"] == sel_month)]
    sub_month_exp = df_exp[(df_exp["year"] == sel_year) & (df_exp["month"] == sel_month)]

    if sub_month.empty:
        st.warning("Нет данных за выбранный период.")
    else:
        period_label = f"{MONTH_NAMES.get(sel_month, '')} {sel_year}"

            # ── stats ─────────────────────────────────────────────────────────────
        st.subheader(f"📊 {period_label}")
        st.markdown(f"""
        <div style="background:white;padding:24px;border-radius:12px;box-shadow:0 2px 10px rgba(30,37,87,0.12);border-left:4px solid #e8832a;">
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;text-align:center;">
            <div>
              <div style="font-size:2em;font-weight:900;color:#1e2557;">{len(sub_month)}</div>
              <div style="font-size:0.78em;color:#888;">Отзывов</div>
            </div>
            <div>
              <div style="font-size:2em;font-weight:900;color:#e8832a;">{sub_month['rating'].mean():.1f} ★</div>
              <div style="font-size:0.78em;color:#888;">Средний рейтинг</div>
            </div>
            <div>
              <div style="font-size:2em;font-weight:900;color:#39dfb6;">{(sub_month["sentiment"]=="positive").sum()}</div>
              <div style="font-size:0.78em;color:#888;">Позитивных</div>
            </div>
            <div>
              <div style="font-size:2em;font-weight:900;color:#fe7070;">{(sub_month["sentiment"]=="negative").sum()}</div>
              <div style="font-size:0.78em;color:#888;">Негативных</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        top_cats_month = sub_month_exp["category"].value_counts().head(6)

        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;">
          <div style="font-weight:700;font-size:0.95em;color:#1e2557;padding:6px 0 4px;">Сентимент месяца:</div>
          <div style="font-weight:700;font-size:0.95em;color:#1e2557;padding:6px 0 4px;">Топ категорий месяца:</div>
        </div>
        """, unsafe_allow_html=True)
        col_l, col_r = st.columns([0.5, 0.5])
        with col_l:
            sent_m     = sub_month["sentiment"].value_counts().rename(index=SENT_LABELS)
            color_map_ru = {"Позитивный":"#39dfb6","Негативный":"#fe7070","Нейтральный":"#ffda3b"}
            fig, ax = plt.subplots(figsize=(4, 4))
            pie_colors = [color_map_ru.get(s, "#e8832a") for s in sent_m.index]
            ax.pie(sent_m, labels=sent_m.index, colors=pie_colors, autopct="%1.0f%%", startangle=90,
                   textprops={"fontsize": 8})
            plt.tight_layout()
            st.pyplot(fig)
        with col_r:
            fig, ax = plt.subplots(figsize=(4, 4))
            top_cats_month.sort_values().plot(kind="barh", ax=ax, color="#e8832a")
            ax.set_xlabel("Кол-во отзывов", fontsize=8)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        st.divider()

        # ── AI Insight ────────────────────────────────────────────────────────
        st.subheader(f"🧠 AI-инсайт — {period_label}")
        month_texts  = sub_month["text"].tolist()
        month_kws    = top_keywords(month_texts, n=10)
        top_cat_list = top_cats_month.index.tolist()
        pos_pct  = round((sub_month["sentiment"]=="positive").mean()*100, 1)
        neg_pct  = round((sub_month["sentiment"]=="negative").mean()*100, 1)
        neu_pct  = round(100 - pos_pct - neg_pct, 1)
        sent_summary = f"Позитивных: {pos_pct}%, Негативных: {neg_pct}%, Нейтральных: {neu_pct}%"

        if month_kws:
            st.markdown("**Ключевые слова периода:** " +
                        " ".join([f'<span class="tag">{k}</span>' for k in month_kws]),
                        unsafe_allow_html=True)

        month_key = f"month_insight_{sel_year}_{sel_month}"
        if st.button("🚀 Сгенерировать инсайт по периоду", key="btn_month_insight"):
            with st.spinner(f"Анализирую {period_label}..."):
                result = call_groq(format_month_prompt(
                    period_label, month_texts, month_kws, top_cat_list, sent_summary
                ), max_tokens=1800)
            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
            else:
                st.session_state[month_key] = result
        if month_key in st.session_state:
            result    = st.session_state[month_key]
            pos_html  = "".join([f'<p class="like-item">✅ {x}</p>'    for x in result.get("positive_highlights",[])])
            neg_html  = "".join([f'<p class="dislike-item">❌ {x}</p>' for x in result.get("negative_highlights",[])])
            tags_html = "".join([f'<span class="tag">{t}</span>'       for t in result.get("hot_topics",[])])
            st.markdown(f"""
            <div class="insight-box">
              <div class="insight-headline">📌 {result.get("headline","")}</div>
              <hr style="border:none;border-top:1px solid #eef0f8;margin:8px 0;">

              <div class="insight-section-title">🔥 Горячие темы</div>
              <div style="margin-bottom:10px;">{tags_html}</div>

              <div class="insight-section-title">💙 Что хвалили</div>
              {pos_html}

              <div class="insight-section-title">💔 На что жаловались</div>
              {neg_html}

              <div class="insight-section-title">🧠 Нарратив периода</div>
              <div class="insight-deep">{result.get("insight","")}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Review cards for selected month ───────────────────────────────────
        st.subheader(f"💬 Отзывы за {period_label}")
        render_review_cards(sub_month, key_prefix="timeline")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: СРАВНЕНИЕ СДУ vs НАРХОЗ
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Сравнение":
    st.title("🏆 СДУ vs Нархоз")
    st.markdown(
        '<p style="color:#888;font-size:0.85em;margin-top:-10px;margin-bottom:18px;">' +
        'Сравнение по данным 2GIS' +
        '</p>',
        unsafe_allow_html=True,
    )

    dn = load_narxoz()

    # ── Ключевые метрики side by side ─────────────────────────────────────────
    def uni_stats(d, name):
        return {
            "name":        name,
            "total":       len(d),
            "avg_rating":  round(d["rating"].mean(), 2),
            "pct_pos":     round((d.sentiment=="positive").mean()*100, 1),
            "pct_neg":     round((d.sentiment=="negative").mean()*100, 1),
            "pct_reply":   round((d.has_official_reply==True).mean()*100, 1),
        }

    s_sdu    = uni_stats(df,  "СДУ")
    s_narxoz = uni_stats(dn, "Нархоз")

    SDU_COLOR    = "#1e2557"
    NARXOZ_COLOR = "#e8832a"

    # Header cards
    h1, h2 = st.columns(2)

    for col, s, color in [(h1, s_sdu, SDU_COLOR), (h2, s_narxoz, NARXOZ_COLOR)]:
        col.markdown(f"""
            <div class="university-card"
                style="background:{color};border-radius:14px;padding:20px 24px;
                        text-align:center;margin-bottom:16px;">

            <div style="font-size:1.4em;font-weight:900;">
                {s["name"]}
            </div>

            <div style="font-size:0.85em;opacity:0.8;">
                {s["total"]} отзывов
            </div>

            </div>
            """, unsafe_allow_html=True)
    st.divider()

    # ── Метрики сравнения ────────────────────────────────────────────────────
    metrics = [
        ("Средний рейтинг ★", "avg_rating", False),
        ("Позитивных %",      "pct_pos",    False),
        ("Негативных %",      "pct_neg",    True),   # True = lower is better
        ("% с ответом",       "pct_reply",  False),
    ]

    cols = st.columns(len(metrics))
    for col, (label, key, lower_is_better) in zip(cols, metrics):
        v_sdu    = s_sdu[key]
        v_narxoz = s_narxoz[key]
        if lower_is_better:
            winner = "СДУ" if v_sdu < v_narxoz else ("Нархоз" if v_narxoz < v_sdu else "—")
        else:
            winner = "СДУ" if v_sdu > v_narxoz else ("Нархоз" if v_narxoz > v_sdu else "—")
        win_color = SDU_COLOR if winner == "СДУ" else (NARXOZ_COLOR if winner == "Нархоз" else "#888")

        col.markdown(f"""
        <div style="background:white;border-radius:12px;padding:14px;
                    text-align:center;box-shadow:0 1px 8px rgba(30,37,87,0.08);
                    margin-bottom:12px;">
          <div style="font-size:0.78em;color:#888;margin-bottom:8px;">{label}</div>
          <div style="display:flex;justify-content:space-around;align-items:center;">
            <div>
              <div style="font-size:1.5em;font-weight:900;color:{SDU_COLOR};">{v_sdu}</div>
              <div style="font-size:0.7em;color:#aaa;">СДУ</div>
            </div>
            <div style="font-size:1.2em;color:#ccc;">vs</div>
            <div>
              <div style="font-size:1.5em;font-weight:900;color:{NARXOZ_COLOR};">{v_narxoz}</div>
              <div style="font-size:0.7em;color:#aaa;">Нархоз</div>
            </div>
          </div>
          <div style="margin-top:8px;font-size:0.78em;font-weight:700;color:{win_color};">
            {"🏆 " + winner if winner != "—" else "—"}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Распределение рейтингов ──────────────────────────────────────────────
    st.subheader("Распределение рейтингов")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=False)
    for ax, d, name, color in [
        (axes[0], df,  "СДУ",    SDU_COLOR),
        (axes[1], dn, "Нархоз", NARXOZ_COLOR),
    ]:
        counts = [len(d[d["rating"]==r]) for r in [1,2,3,4,5]]
        bars = ax.bar([1,2,3,4,5], counts, color=color, edgecolor="none", width=0.6, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(cnt), ha="center", va="bottom", fontsize=8, color="#1e2557")
        ax.set_title(name, fontweight="bold", color=color)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlabel("Рейтинг"); ax.set_ylabel("Кол-во")
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Динамика рейтинга по месяцам ────────────────────────────────────────
    st.subheader("Динамика среднего рейтинга по месяцам")
    timeline_sdu    = df.groupby("year_month")["rating"].mean()
    timeline_narxoz = dn.groupby("year_month")["rating"].mean()
    all_months = sorted(set(timeline_sdu.index) | set(timeline_narxoz.index))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timeline_sdu.index,    timeline_sdu.values,    label="СДУ",
            color=SDU_COLOR,    linewidth=2.5, marker="o", markersize=4)
    ax.plot(timeline_narxoz.index, timeline_narxoz.values, label="Нархоз",
            color=NARXOZ_COLOR, linewidth=2.5, marker="o", markersize=4)
    ax.set_ylim(1, 5.5)
    ax.set_ylabel("Средний рейтинг")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(); ax.grid(alpha=0.2); plt.xticks(rotation=30); plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Сентимент сравнение ──────────────────────────────────────────────────
    st.subheader("Сентимент")
    sent_data = {
        "Позитивный": [s_sdu["pct_pos"], s_narxoz["pct_pos"]],
        "Нейтральный": [
            round((df.sentiment=="neutral").mean()*100, 1),
            round((dn.sentiment=="neutral").mean()*100, 1),
        ],
        "Негативный": [s_sdu["pct_neg"], s_narxoz["pct_neg"]],
    }
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(2)
    sent_colors = {"Позитивный": "#39dfb6", "Нейтральный": "#ffda3b", "Негативный": "#fe7070"}
    for sent, vals in sent_data.items():
        bars = ax.bar(x, vals, bottom=bottom, label=sent, color=sent_colors[sent], width=0.5)
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_y()+bar.get_height()/2,
                        f"{val:.0f}%", ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(["СДУ", "Нархоз"], fontweight="bold")
    ax.set_ylabel("% отзывов"); ax.set_ylim(0, 105)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8); plt.tight_layout()
    st.pyplot(fig)

    # ── Проблемные зоны (топ негативных категорий с цитатами) ────────────────
    st.subheader("🚨 Проблемные зоны")
    st.caption("Категории с наибольшим числом негативных отзывов. Нажмите на цитату для просмотра полного отзыва.")

    # ── White/red style for quote buttons + wide dialog ─────────────────────
    st.markdown("""
    <style>
    /* Widen the dialog modal */
    div[data-testid="stDialog"] > div > div {
        max-width: 780px !important;
        width: 90vw !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Modal dialog for full review ─────────────────────────────────────────
    @st.dialog("💬 Полный отзыв")
    def show_full_review(text, rating, reactions):
        stars = "★" * int(rating) + "☆" * (5 - int(rating))
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:14px;align-items:center;">
          <span style="font-size:1.1em;color:#c0392b;font-weight:700;">{stars}</span>
          <span style="background:#fee2e2;color:#c0392b;border-radius:20px;
                       padding:2px 10px;font-size:0.82em;font-weight:700;">★ {rating}</span>
          <span style="background:#fff3f3;color:#e67e22;border-radius:20px;
                       padding:2px 10px;font-size:0.82em;font-weight:700;">👍 {reactions} реакций</span>
        </div>
        <div style="border-left:4px solid #fe7070;padding:14px 18px;background:#fff5f5;
                    border-radius:8px;font-size:0.97em;color:#1e2557;line-height:1.7;
                    white-space:pre-wrap;">{text}</div>
        """, unsafe_allow_html=True)

    neg_df = df_exp[df_exp["sentiment"] == "negative"]
    if len(neg_df) > 0:
        neg_by_cat = neg_df.groupby("category").size().sort_values(ascending=False)
        skip_cats = {"Нерелевантный / пустой отзыв", "Общий негативный отзыв"}
        neg_by_cat = neg_by_cat[~neg_by_cat.index.isin(skip_cats)].head(5)

        used_review_ids = set()

        for rank, (cat, cnt) in enumerate(neg_by_cat.items(), 1):
            cat_negs = neg_df[neg_df["category"] == cat].copy()
            cat_negs_unique = cat_negs[~cat_negs["orig_idx"].isin(used_review_ids)]

            if len(cat_negs_unique) == 0:
                continue

            # Show reviews where this category is the PRIMARY (first) one
            primary_mask = df.loc[cat_negs_unique["orig_idx"], "categories_raw"].apply(
                lambda x: strip_cat_num(str(x).split(";")[0].strip()) == cat
            )
            primary_idx = primary_mask[primary_mask].index.tolist()
            primary_rows = cat_negs_unique[cat_negs_unique["orig_idx"].isin(primary_idx)]

            # Fall back to all negatives if no primary-tagged reviews found
            pool = primary_rows if len(primary_rows) > 0 else cat_negs_unique

            top_rows = pool.sort_values("reactions_total", ascending=False).head(2)
            used_review_ids.update(top_rows["orig_idx"].tolist())

            avg_cat_r = cat_negs["rating"].mean()
            total_reactions = int(cat_negs["reactions_total"].sum())

            # ── Category header ───────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px 20px;
                        margin-top:12px;margin-bottom:6px;
                        box-shadow:0 1px 8px rgba(30,37,87,0.08);
                        border-left:4px solid #fe7070;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-weight:700;font-size:1em;color:#1e2557;">{rank}. {cat}</div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                  <span style="background:#fee2e2;color:#c0392b;border-radius:20px;
                               padding:2px 10px;font-size:0.82em;font-weight:700;">
                    {cnt} негативных</span>
                  <span style="background:#f0f2fa;color:#1e2557;border-radius:20px;
                               padding:2px 10px;font-size:0.82em;font-weight:700;">
                    ★ {avg_cat_r:.1f}</span>
                  <span style="background:#fff3f3;color:#e67e22;border-radius:20px;
                               padding:2px 10px;font-size:0.82em;font-weight:700;">
                    👍 {total_reactions} реакций</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Quote buttons → open modal ────────────────────────────────────
            for _, row in top_rows.iterrows():
                q = row["text"]
                if not isinstance(q, str) or len(q.strip()) <= 5:
                    continue
                preview = q[:160] + ("…" if len(q) > 160 else "")
                btn_key = f"rev_{rank}_{row['orig_idx']}"
                with st.container():
                    st.markdown('<div class="quote-btn">', unsafe_allow_html=True)
                    if st.button(f'💬 "{preview}"', key=btn_key, use_container_width=True):
                        show_full_review(q, row["rating"], row["reactions_total"])
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    else:
        st.info("Негативных отзывов не найдено.")