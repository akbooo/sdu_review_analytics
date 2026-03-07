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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ── download stopwords if needed ──────────────────────────────────────────────
try:
    stopwords.words("russian")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ── config ────────────────────────────────────────────────────────────────────
MODEL        = "llama-3.1-8b-instant"
CSV_PATH     = "sdu_overall_march_classified.csv"

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
.stButton>button:hover { background-color:#1e2557 !important; transition:0.2s; }
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

def parse_date(x):
    if pd.isna(x): return None
    parts = str(x).split()
    if len(parts) != 3: return None
    d, m, y = parts
    m = MONTHS_RU.get(m.lower())
    return f"{y}-{m}-{d.zfill(2)}" if m else None

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
    df["categories_raw"]  = df["categories"].astype(str).str.strip()
    df["category"]        = df["categories_raw"].str.split(";").str[0].str.strip().apply(strip_cat_num)
    df["date_str"]        = df["date"].apply(parse_date)
    df["date"]            = pd.to_datetime(df["date_str"], errors="coerce")
    df["year"]            = df["date"].dt.year
    df["month"]           = df["date"].dt.month
    df["year_month"]      = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year_month_str"]  = df["date"].dt.to_period("M").astype(str)
    df = df.reset_index(drop=True)
    df["orig_idx"] = df.index
    return df

@st.cache_data
def make_exploded(df):
    rows = []
    for _, row in df.iterrows():
        cats = [strip_cat_num(c.strip()) for c in str(row["categories_raw"]).split(";") if c.strip()]
        for cat in cats:
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
    for row in dataframe.head(show_n).itertuples():
        txt = row.text[:400] + ("..." if len(row.text) > 400 else "")
        cats_display = " · ".join([strip_cat_num(c.strip()) for c in row.categories_raw.split(";") if c.strip()])
        sent_color = {"positive":"#39dfb6","negative":"#fe7070","neutral":"#ffbe51"}.get(row.sentiment, "#ccc")
        st.markdown(f"""
        <div class="review-card" style="border-left:4px solid {sent_color};">
            <p style="margin:0;color:#1e2557;">{txt}</p>
            <p class="review-meta">
              ⭐ {row.rating} &nbsp;|&nbsp; 👍 {row.reactions_total} реакций
              &nbsp;|&nbsp; <span style="color:#e8832a;">{cats_display}</span>
            </p>
        </div>""", unsafe_allow_html=True)

# ── patch sidebar collapse button text via JS ─────────────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
function patchSidebarButton() {
  const doc = window.parent.document;
  // Find all spans that contain material icon text
  doc.querySelectorAll('span').forEach(s => {
    const t = s.textContent.trim();
    if (t === 'keyboard_double_arrow_right' || t === 'keyboard_double_arrow_left') {
      s.textContent = t === 'keyboard_double_arrow_right' ? '▶' : '◀';
      s.style.fontFamily = 'Arial, sans-serif';
      s.style.fontSize = '16px';
      s.style.fontWeight = 'bold';
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
    st.markdown(
        '<p style="color:#888;font-size:0.85em;margin-top:-10px;margin-bottom:18px;">'
        '🕐 Данные последний раз обновлены: <strong style="color:#e8832a;">5 марта 2026</strong>'
        '</p>',
        unsafe_allow_html=True,
    )
    # ── Executive Summary ─────────────────────────────────────────────────────
    avg_r     = df["rating"].mean()
    pct_pos   = (df.sentiment == "positive").mean() * 100
    pct_neg   = (df.sentiment == "negative").mean() * 100
    pct_reply = (df.has_official_reply == True).mean() * 100

    df_sorted = df.dropna(subset=["year_month"]).sort_values("year_month")
    months_all = sorted(df_sorted["year_month"].unique())
    half = max(1, len(months_all) // 2)
    avg_recent  = df_sorted[df_sorted["year_month"].isin(months_all[-half:])]["rating"].mean()
    avg_earlier = df_sorted[df_sorted["year_month"].isin(months_all[:half])]["rating"].mean()
    trend_delta = avg_recent - avg_earlier
    trend_arrow = "📈" if trend_delta > 0.1 else ("📉" if trend_delta < -0.1 else "➡️")
    trend_label = "рост" if trend_delta > 0.1 else ("снижение" if trend_delta < -0.1 else "стабильно")

    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:24px 28px;
                box-shadow:0 2px 12px rgba(30,37,87,0.10);
                border-left:6px solid #e8832a;margin-bottom:20px;">
      <div style="font-size:1.1em;font-weight:800;color:#1e2557;margin-bottom:18px;">
        🎯 Ключевые показатели
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px;text-align:center;">
        <div>
          <div style="font-size:2em;font-weight:900;color:#e8832a;">{avg_r:.2f}</div>
          <div style="font-size:0.78em;color:#888;">Средний рейтинг</div>
        </div>
        <div>
          <div style="font-size:2em;font-weight:900;color:#39dfb6;">{pct_pos:.0f}%</div>
          <div style="font-size:0.78em;color:#888;">Позитивных</div>
        </div>
        <div>
          <div style="font-size:2em;font-weight:900;color:#fe7070;">{pct_neg:.0f}%</div>
          <div style="font-size:0.78em;color:#888;">Негативных</div>
        </div>
        <div>
          <div style="font-size:2em;font-weight:900;color:#1e2557;">{len(df)}</div>
          <div style="font-size:0.78em;color:#888;">Всего отзывов</div>
        </div>
        <div>
          <div style="font-size:1.6em;font-weight:900;color:#1e2557;">{trend_arrow} {abs(pct_reply):.0f}%</div>
          <div style="font-size:0.78em;color:#888;">Отвечено</div>
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
    st.subheader("Статистика категории")
    m1, m2, m3 = st.columns(3)
    with m1: metric_card("Отзывов", len(orig_idxs))
    with m2: metric_card("Доля", f"{percent:.1f}%")
    with m3: metric_card("Средний рейтинг", f"{sub_exp['rating'].mean():.1f} ★")

    st.markdown("**Ключевые слова:**")
    if keywords:
        st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in keywords]),
                    unsafe_allow_html=True)
    else:
        st.write("—")

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

    st.subheader("🗓️ Выберите период")
    col_y, col_m = st.columns(2)
    with col_y:
        years    = sorted(df_dated["year"].dropna().unique().astype(int), reverse=True)
        sel_year = st.selectbox("📅 Год", years, key="tl_year")
    with col_m:
        months_for_year = sorted(
            df_dated[df_dated["year"] == sel_year]["month"].dropna().unique().astype(int)
        )
        sel_month = st.selectbox(
            "🗓 Месяц", months_for_year, key="tl_month",
            format_func=lambda m: MONTH_NAMES.get(m, str(m))
        )

    sub_month     = df[(df["year"] == sel_year) & (df["month"] == sel_month)]
    sub_month_exp = df_exp[(df_exp["year"] == sel_year) & (df_exp["month"] == sel_month)]

    if sub_month.empty:
        st.warning("Нет данных за выбранный период.")
    else:
        period_label = f"{MONTH_NAMES.get(sel_month,'')} {sel_year}"

        # ── stats ─────────────────────────────────────────────────────────────
        st.subheader(f"📊 {period_label}")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: metric_card("Отзывов", len(sub_month))
        with mc2: metric_card("Средний рейтинг", f"{sub_month['rating'].mean():.1f} ★")
        with mc3: metric_card("Позитивных", str((sub_month["sentiment"]=="positive").sum()))
        with mc4: metric_card("Негативных", str((sub_month["sentiment"]=="negative").sum()))

        top_cats_month = sub_month_exp["category"].value_counts().head(6)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Сентимент месяца:**")
            sent_m     = sub_month["sentiment"].value_counts().rename(index=SENT_LABELS)
            color_map_ru = {"Позитивный":"#39dfb6","Негативный":"#fe7070","Нейтральный":"#ffda3b"}
            fig, ax    = plt.subplots(figsize=(4, 4))
            pie_colors = [color_map_ru.get(s, "#e8832a") for s in sent_m.index]
            ax.pie(sent_m, labels=sent_m.index, colors=pie_colors, autopct="%1.0f%%", startangle=90)
            st.pyplot(fig)
        with col_r:
            st.markdown("**Топ категорий месяца:**")
            fig, ax = plt.subplots(figsize=(6, 3))
            top_cats_month.sort_values().plot(kind="barh", ax=ax, color="#e8832a")
            ax.set_xlabel("Кол-во отзывов"); plt.tight_layout()
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
    st.caption("Категории с наибольшим числом негативных отзывов и реальные цитаты студентов")

    neg_df = df_exp[df_exp["sentiment"] == "negative"]
    if len(neg_df) > 0:
        neg_by_cat = neg_df.groupby("category").size().sort_values(ascending=False)
        # exclude irrelevant categories
        skip_cats = {"Нерелевантный / пустой отзыв", "Общий негативный отзыв"}
        neg_by_cat = neg_by_cat[~neg_by_cat.index.isin(skip_cats)].head(5)

        for rank, (cat, cnt) in enumerate(neg_by_cat.items(), 1):
            cat_negs = neg_df[neg_df["category"] == cat].copy()
            # pick top 2 most-reacted reviews as quotes
            quotes = (cat_negs.sort_values("reactions_total", ascending=False)
                              .head(2)["text"]
                              .tolist())
            avg_cat_r = cat_negs["rating"].mean()

            quotes_html = "".join([
                f'<div style="border-left:3px solid #fe7070;padding:6px 12px;margin:6px 0;' +
                f'background:#fff5f5;border-radius:0 8px 8px 0;font-size:0.88em;color:#444;">' +
                f'"{ q[:160] }{"..." if len(q)>160 else ""}"</div>'
                for q in quotes if isinstance(q, str) and len(q.strip()) > 5
            ])

            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px 20px;
                        margin-bottom:12px;box-shadow:0 1px 8px rgba(30,37,87,0.08);
                        border-left:4px solid #fe7070;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div style="font-weight:700;font-size:1em;color:#1e2557;">
                  {rank}. {cat}
                </div>
                <div style="display:flex;gap:12px;">
                  <span style="background:#fee;color:#c0392b;border-radius:20px;
                               padding:2px 10px;font-size:0.82em;font-weight:700;">
                    {cnt} негативных
                  </span>
                  <span style="background:#f0f2fa;color:#1e2557;border-radius:20px;
                               padding:2px 10px;font-size:0.82em;font-weight:700;">
                    ★ {avg_cat_r:.1f}
                  </span>
                </div>
              </div>
              {quotes_html}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Негативных отзывов не найдено.")

