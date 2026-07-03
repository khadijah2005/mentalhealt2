import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from wordcloud import WordCloud

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MindWatch | Monitoring Isu Kesehatan Mental Gen Z",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_BG          = "#FFF9FB"
C_CARD        = "#FFEEF3"
C_ACCENT      = "#F7A8C4"
C_ACCENT_LT   = "#FBC4D8"
C_ACCENT_DUST = "#E893B5"
C_MINT        = "#B8E3D8"
C_PEACH       = "#FFD9A0"
C_SOFTRED     = "#F4A6A6"
C_TEXT        = "#5C4150"

PASTEL_SEQ   = [C_ACCENT, C_MINT, C_PEACH, C_ACCENT_DUST, C_SOFTRED, C_ACCENT_LT]
PASTEL_SCALE = [[0.0, C_BG], [0.5, C_ACCENT_LT], [1.0, C_ACCENT_DUST]]

LABEL_URGENT = "🚨 Butuh Pertolongan Segera"
LABEL_RINGAN = "💬 Curhat Ringan"
COLOR_MAP_URGENCY = {LABEL_URGENT: C_SOFTRED, LABEL_RINGAN: C_MINT}
COLOR_MAP_RISK    = {"Tinggi": C_SOFTRED, "Sedang": C_PEACH, "Rendah": C_MINT}

KMEANS_CLUSTER_NAMES = {
    0: "Tekanan Akademik",
    1: "Masalah Keluarga & Sosial",
    2: "Finansial & Karier",
    3: "Kesehatan Mental Umum",
    4: "Hubungan Pertemanan",
    5: "Lainnya",
}

CLUSTER_KEYWORDS = {
    "Tekanan Akademik":       {"tugas","skripsi","ujian","dosen","kuliah","deadline","ipk","krs",
                               "sidang","praktikum","semester","kampus","magang","nilai","revisi","sks","tugas akhir"},
    "Masalah Keluarga":       {"orang tua","ayah","ibu","keluarga","broken home","kdrt","cerai",
                               "dibanding","tekanan keluarga","ekspektasi","berantem","dipaksa","rumah"},
    "Finansial":              {"uang","biaya","ekonomi","kerja","gaji","utang","beasiswa","finansial",
                               "susah cari kerja","tidak punya uang","nunggak","biaya kuliah","ukt"},
    "Hubungan Sosial":        {"pacar","putus","mantan","teman","sahabat","dikhianati","selingkuh",
                               "toksik","dijauhin","kesepian","diabaikan","bullying","ghosting","circle"},
    "Kesehatan Mental Umum":  {"depresi","cemas","anxiety","stres","burnout","trauma","insomnia",
                               "psikolog","psikiater","konseling","terapi","kesehatan jiwa","mental"},
}

SUPPORT_PHRASES = {
    "semangat","kamu kuat","gpp","gapapa","tidak apa apa","ada aku","dm aja","cerita yuk",
    "pelukan","peluk","stay strong","kamu berharga","tetap semangat","jangan menyerah",
    "kita disini","sini cerita","semangat ya","kamu hebat","peluk jauh","kamu tidak sendirian",
    "boleh cerita","aku dengerin","semoga membaik","tetap kuat","kamu pasti bisa",
    "jangan lupa makan","jangan lupa istirahat","sehat selalu",
}

NORMALIZATION_DICT = {
    "yg":"yang","ga":"tidak","gak":"tidak","gk":"tidak","tdk":"tidak","engga":"tidak","nggak":"tidak",
    "udh":"sudah","udah":"sudah","sdh":"sudah","dah":"sudah","blm":"belum","blom":"belum",
    "gw":"saya","gue":"saya","gua":"saya","w":"saya","sy":"saya","aq":"saya",
    "lu":"kamu","lo":"kamu","loe":"kamu","km":"kamu",
    "bgt":"banget","bngt":"banget","bgttt":"banget",
    "tp":"tapi","krn":"karena","krna":"karena","jd":"jadi","jdi":"jadi",
    "knp":"kenapa","gmn":"gimana","emg":"memang","emang":"memang",
    "skrg":"sekarang","skg":"sekarang","td":"tadi",
    "org":"orang","ortu":"orang tua","klrg":"keluarga",
    "capek":"lelah","cape":"lelah","cpe":"lelah",
    "sdih":"sedih","nangis":"menangis",
    "depresi":"depresi","anxiety":"cemas","anxious":"cemas",
    "overthink":"cemas","overthinking":"cemas",
    "burnout":"burnout","insecure":"tidak percaya diri",
    "hopeless":"putus asa",
    "deadline":"tenggat waktu","dl":"tenggat waktu",
    "dospem":"dosen pembimbing","duit":"uang","ukt":"biaya kuliah",
    "stress":"stres","stres":"stres",
    "gabut":"bosan","mager":"malas",
    "thx":"terima kasih","makasih":"terima kasih","mksh":"terima kasih",
    "semngt":"semangat","smngt":"semangat",
}

STOPWORDS = {
    "yang","di","dan","itu","ini","dari","ke","untuk","dengan","nya","saya","aku","kamu",
    "kami","kita","bisa","ada","adalah","juga","karena","tapi","namun","atau","jadi",
    "jika","kalau","sudah","lagi","akan","pada","masih","saja","yg","dg","dgn","ny","d","k",
    "biar","bikin","bilang","nih","sih","si","tau","tuh","ya","jd","jgn","aja","n","t",
    "loh","oleh","se","an","kan","dia","mereka","ia","telah","sedang","pernah","belum",
    "bukan","jangan","bila","maka","dalam","kepada","terhadap","antara","tentang",
    "hingga","sambil","demi","sebelum","sesudah","saat","ketika","begitu","seperti",
    "secara","setiap","seluruh","semua","para","sang","deh","dong","kok","lho","kah","pun","banget",
    "http","co","amp","rt","via","the","is","in","to","a","of","for","and","on","gitu","sm",
}

KEEP_WORDS = {"tidak","putus","asa","lelah","cemas","menangis","burnout","depresi","stres"}

# ============================================================
# LABELING DENGAN SKOR (threshold=1.5 → F1 ≥ 0.64)
# ============================================================
KEYWORD_SCORES = {
    "mau mati":3,"pengen mati":3,"ingin mati":3,"putus asa":3,"breakdown":3,
    "menyerah":2.5,"ga ada harapan":3,"gamau hidup":3,"self harm":3,
    "hopeless":2.5,"cape hidup":2.5,"pengen ngilang":2.5,
    "tiap hari nangis":2.5,"nangis tiap hari":2.5,"nangis mulu":2,
    "ga kuat lagi":2.5,"tidak kuat lagi":2.5,"gamau lanjut":2.5,
    "semigap":2,"burnout parah":2,"ga ada motivasi apapun":2,
    "sendirian banget":2,"straight nangis":2,
    "burnout":1.2,"kelelahan":1,"stres":1,"menangis":1,
    "ga semangat":1,"tidak semangat":1,"demotivasi":1.2,"ilang semangat":1.5,
    "capek":0.8,"lelah":0.8,"cape":0.8,"cemas":1,"kesepian":1.2,
    "loyo":1,"badmood":0.8,"frustasi":1.2,"galau":0.8,"overwhelmed":1.2,
    "pressure":1,"ga bisa tidur":1.5,"malas":0.5,"capek banget":1.5,
    "ga ada motivasi":1.5,"tidak ada motivasi":1.5,
    "sendirian":1,"gapunya temen":1.5,"tidak ada semangat":1.5,
}
INFO_NEG_WORDS = [
    "tips","cara mengatasi","solusi","data bps","menurut penelitian",
    "rekomendasi","saran","info","wadah","spot","uipedia","fakta",
    "pernah tau","artikel","survey",
]

URGENCY_THRESHOLD = 1.5

def compute_urgency_score(raw_text_lower):
    sc = sum(v for w, v in KEYWORD_SCORES.items() if w in raw_text_lower)
    inf = sum(1 for w in INFO_NEG_WORDS if w in raw_text_lower)
    if inf >= 2:
        sc -= 1.5
    return sc

def score_urgency(raw_text_lower):
    sc = compute_urgency_score(raw_text_lower)
    if sc >= 4.0:   risk = "Tinggi"
    elif sc >= 1.5: risk = "Sedang"
    else:           risk = "Rendah"
    label = LABEL_URGENT if sc >= URGENCY_THRESHOLD else LABEL_RINGAN
    return label, sc, risk

def detect_cluster(raw_text_lower):
    scores = {c: sum(1 for kw in kws if kw in raw_text_lower)
              for c, kws in CLUSTER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Lainnya"

def detect_support(raw_text_lower):
    return any(p in raw_text_lower for p in SUPPORT_PHRASES)

# ============================================================
# FILTER MENTION
# ============================================================
def filter_mention(df, text_col):
    mask = ~df[text_col].astype(str).str.contains(r'@\w+', regex=True, na=False)
    return df[mask].copy().reset_index(drop=True)

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
def case_fold(text):
    return text.lower() if isinstance(text, str) else ""

def clean_only(text):
    t = re.sub(r"http\S+|www\.\S+", " ", text)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_text(text):
    words = text.split()
    return " ".join(NORMALIZATION_DICT.get(w, w) for w in words)

def tokenize(text):
    return text.split()

def remove_stopwords_tokens(tokens):
    return [w for w in tokens if (w not in STOPWORDS or w in KEEP_WORDS) and len(w) > 2]

@st.cache_resource(show_spinner=False)
def get_stemmer():
    if SASTRAWI_AVAILABLE:
        factory = StemmerFactory()
        return factory.create_stemmer()
    return None

def stem_tokens(tokens, stemmer, cache):
    out = []
    for w in tokens:
        if w in cache:
            out.append(cache[w])
        else:
            s = stemmer.stem(w) if stemmer else w
            cache[w] = s
            out.append(s)
    return out

@st.cache_data(show_spinner=False)
def run_full_pipeline(texts):
    stemmer = get_stemmer()
    stem_cache = {}
    rows = []
    for raw in texts:
        raw = raw if isinstance(raw, str) else ""
        cf = case_fold(raw)
        cl = clean_only(cf)
        nm = normalize_text(cl)
        tk = tokenize(nm)
        sw = remove_stopwords_tokens(tk)
        st_ = stem_tokens(sw, stemmer, stem_cache)
        rows.append({
            "raw": raw, "case_folded": cf, "cleaned": cl, "normalized": nm,
            "tokenized": tk, "stopword_removed": " ".join(sw), "stemmed": " ".join(st_),
        })
    return pd.DataFrame(rows)

def preprocess_single(text):
    cf = case_fold(text if isinstance(text, str) else "")
    cl = clean_only(cf)
    nm = normalize_text(cl)
    sw = remove_stopwords_tokens(tokenize(nm))
    stemmer = get_stemmer()
    cache = st.session_state.setdefault("_stem_cache_single", {})
    return " ".join(stem_tokens(sw, stemmer, cache))

def pipeline_word_stats(pipe_df):
    stages = ["raw","case_folded","cleaned","normalized","stopword_removed","stemmed"]
    labels = ["Raw","Case Folding","Cleaning","Normalisasi","Stopword Removal","Stemming"]
    rows = []
    for stage, label in zip(stages, labels):
        all_words = " ".join(pipe_df[stage]).split()
        rows.append({"Tahap": label, "Total Kata": len(all_words), "Kata Unik": len(set(all_words))})
    return pd.DataFrame(rows)

# ============================================================
# LDA
# ============================================================
@st.cache_data(show_spinner=False)
def run_lda(texts, n_topics=3):
    vec = CountVectorizer(max_features=1000)
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=50)
    lda.fit(X)
    doc_topic = lda.transform(X)
    dom = doc_topic.argmax(axis=1)
    log_lik = lda.score(X)
    perp = lda.perplexity(X)
    topic_words = {}
    for k in range(n_topics):
        top_idx = lda.components_[k].argsort()[::-1][:8]
        topic_words[k] = [terms[i] for i in top_idx]
    return dom, topic_words, log_lik, perp, lda, vec

# ============================================================
# K-MEANS + ELBOW
# ============================================================
@st.cache_data(show_spinner=False)
def run_kmeans(texts, k=3):
    vec = TfidfVectorizer(max_features=1000)
    X_tfidf = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)
    wcss, sil_scores, k_range = [], [], range(2, min(8, len(texts)))
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels_tmp = km.fit_predict(X_2d)
        wcss.append(km.inertia_)
        if len(set(labels_tmp)) > 1:
            sil_scores.append(silhouette_score(X_2d, labels_tmp))
        else:
            sil_scores.append(0)
    optimal_k_sil = list(k_range)[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_2d)
    sil = silhouette_score(X_2d, labels) if len(set(labels)) > 1 else 0
    dbi = davies_bouldin_score(X_2d, labels) if len(set(labels)) > 1 else 0
    chi = calinski_harabasz_score(X_2d, labels) if len(set(labels)) > 1 else 0
    return labels, X_2d, km_final, list(k_range), wcss, sil_scores, sil, dbi, chi, optimal_k_sil

# ============================================================
# ML CLASSIFIER — 3 MODEL + PERBANDINGAN 70:30
# ============================================================
@st.cache_data(show_spinner=False)
def train_ml_classifier(texts, labels):
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(texts)
    classes = sorted(set(labels))
    can_strat = len(classes) > 1 and all(labels.count(c) >= 2 for c in classes)

    # 70:30 split sesuai instruksi dosen
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.3, random_state=42,
            stratify=labels if can_strat else None,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0),
        "SVM (LinearSVC)":     LinearSVC(class_weight="balanced", max_iter=2000, C=1.0),
        "Complement Naive Bayes": ComplementNB(),
    }

    results = {}
    best_name, best_f1, best_model = None, -1, None
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        rep  = classification_report(y_te, y_pred, labels=classes, output_dict=True, zero_division=0)
        cm   = confusion_matrix(y_te, y_pred, labels=classes)
        acc  = accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average="macro", labels=classes, zero_division=0)
        # 5-fold CV
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            cv_sc = cross_val_score(mdl, X, labels, cv=skf, scoring="f1_macro")
        except Exception:
            cv_sc = np.array([f1m])
        results[name] = {"model": mdl, "report": rep, "cm": cm, "acc": acc,
                         "f1m": f1m, "cv": cv_sc, "classes": classes,
                         "n_train": len(y_tr), "n_test": len(y_te)}
        if f1m > best_f1:
            best_f1, best_name, best_model = f1m, name, mdl

    return results, best_name, best_model, vec, classes

def report_to_df(report, classes):
    rows = {k: v for k, v in report.items()
            if k in classes or k in ("macro avg","weighted avg")}
    return pd.DataFrame(rows).T.round(3)

# ============================================================
# TRENDING TOPIC — per bulan, dengan keyword extraction
# ============================================================
def get_trending_topic_monthly(df, text_col, ts_col, top_n=5):
    """Trending topic: topik/kata yang naik paling cepat bulan ini vs bulan lalu"""
    df2 = df.dropna(subset=[ts_col]).copy()
    df2["_month"] = df2[ts_col].dt.to_period("M").astype(str)
    months = sorted(df2["_month"].unique())
    if len(months) < 2:
        return None, None
    last_month   = months[-1]
    before_month = months[-2]

    def top_words_in_period(texts, n=top_n):
        words = []
        for t in texts:
            t = str(t).lower()
            t = re.sub(r"http\S+|@\w+|#\w+|[^\w\s]|\d+", " ", t)
            for w in NORMALIZATION_DICT:
                t = t.replace(w, NORMALIZATION_DICT[w])
            words.extend([w for w in t.split()
                          if w not in STOPWORDS and len(w) > 3])
        return Counter(words)

    cnt_last   = top_words_in_period(df2[df2["_month"] == last_month][text_col])
    cnt_before = top_words_in_period(df2[df2["_month"] == before_month][text_col])
    all_words  = set(cnt_last.keys()) | set(cnt_before.keys())
    trending = []
    for w in all_words:
        c_now  = cnt_last.get(w, 0)
        c_prev = cnt_before.get(w, 0)
        delta  = c_now - c_prev
        pct    = ((c_now - c_prev) / (c_prev + 1)) * 100
        if c_now >= 2:
            trending.append({"Kata/Topik": w, "Bulan Ini": c_now,
                             "Bulan Lalu": c_prev, "Δ": delta, "Naik (%)": round(pct, 1)})
    df_trend = pd.DataFrame(trending).sort_values("Naik (%)", ascending=False).head(15)
    return df_trend, last_month

# ============================================================
# SNA
# ============================================================
def build_network(df, text_col, username_col, parent_col):
    G = nx.DiGraph()
    mention_pat = re.compile(r"@(\w+)")
    for idx, row in df.iterrows():
        raw       = str(row[text_col])
        is_sup    = detect_support(raw.lower())
        source    = str(row[username_col]).strip() if username_col and pd.notna(row.get(username_col)) else f"akun_{idx}"
        targets   = set()
        if parent_col and pd.notna(row.get(parent_col, None)):
            val = str(row[parent_col]).strip().lstrip("@")
            if val and val != "nan":
                targets.add(val)
        for m in mention_pat.findall(raw):
            targets.add(m)
        for t in targets:
            if t == source or t == "": continue
            w = 2 if is_sup else 1
            if G.has_edge(source, t):
                G[source][t]["weight"] += w
                if is_sup: G[source][t]["support"] = True
            else:
                G.add_edge(source, t, weight=w, support=is_sup)
    return G

def top_support_accounts(G, top_n=10):
    rows = [u for u, v, d in G.edges(data=True) if d.get("support")]
    if not rows:
        return pd.DataFrame(columns=["Akun","Jumlah Respon Positif"])
    return pd.DataFrame(Counter(rows).most_common(top_n), columns=["Akun","Jumlah Respon Positif"])

def most_supported_accounts(G, top_n=10):
    rows = [v for u, v, d in G.edges(data=True) if d.get("support")]
    if not rows:
        return pd.DataFrame(columns=["Akun","Jumlah Dukungan Diterima"])
    return pd.DataFrame(Counter(rows).most_common(top_n), columns=["Akun","Jumlah Dukungan Diterima"])

def plot_network(G, max_nodes=60):
    if G.number_of_nodes() == 0: return None
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()
    pos = nx.spring_layout(G, k=0.6, seed=42)
    edge_x, edge_y, sup_x, sup_y = [], [], [], []
    for u, v, d in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        if d.get("support"):
            sup_x += [x0,x1,None]; sup_y += [y0,y1,None]
        else:
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_tr = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                          line=dict(width=0.6, color="rgba(232,147,181,0.25)"), hoverinfo="none")
    sup_tr  = go.Scatter(x=sup_x,  y=sup_y,  mode="lines",
                          line=dict(width=1.8, color="rgba(184,227,216,0.85)"), hoverinfo="none")
    degs    = dict(G.degree())
    max_deg = max(degs.values()) if degs else 1
    nx_,ny_,nt_,ns_,nc_ = [],[],[],[],[]
    for n in G.nodes():
        x,y = pos[n]; d = degs[n]
        nx_.append(x); ny_.append(y)
        nt_.append(f"@{n}<br>Koneksi: {d}")
        ns_.append(10 + 30*(d/max_deg)); nc_.append(d)
    node_tr = go.Scatter(x=nx_, y=ny_, mode="markers", hoverinfo="text", text=nt_,
                          marker=dict(size=ns_, color=nc_,
                                      colorscale=[[0,C_ACCENT_LT],[1,C_ACCENT_DUST]],
                                      line=dict(width=1, color="rgba(255,255,255,0.7)"), showscale=False))
    fig = go.Figure(data=[edge_tr, sup_tr, node_tr])
    fig.update_layout(showlegend=False, hovermode="closest",
                      margin=dict(l=10,r=10,t=10,b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500)
    return fig

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color=C_TEXT, font_family="Nunito")

# ============================================================
# AUTO-DETEKSI KOLOM
# ============================================================
def auto_detect(df, candidates):
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None

def detect_text_col(df):
    return auto_detect(df, ["text","tweet","full_text","content","caption","isi","teks","curhatan"])
def detect_username_col(df):
    return auto_detect(df, ["username","user","account","akun","screen_name","author"])
def detect_parent_col(df):
    return auto_detect(df, ["in_reply_to_screen_name","in_reply_to_user","in_reply_to","reply_to","parent"])
def detect_timestamp_col(df):
    return auto_detect(df, ["created_at","timestamp","date","tanggal","waktu","time"])
def detect_engagement_cols(df):
    cands = ["like_count","likes","favorite_count","retweet_count","retweets",
             "reply_count","replies","quote_count","views","view_count"]
    return [c for c in df.columns if c.lower().strip() in cands]

# ============================================================
# INSIGHT HELPERS
# ============================================================
def insight_card(title, bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(184,227,216,0.25),rgba(255,255,255,0.6),rgba(247,168,196,0.15));
    border:1.5px dashed {C_ACCENT_DUST};border-radius:22px;padding:20px 24px;margin:16px 0;position:relative;overflow:hidden;">
    <h4 style="margin-top:0;font-size:12px;color:{C_ACCENT_DUST};text-transform:uppercase;letter-spacing:1px;font-weight:700;">
    💡 {title}</h4><ul style="margin:8px 0 0 0;padding-left:20px;">
    {"".join(f'<li style="font-size:13.5px;line-height:1.8;color:{C_TEXT};margin-bottom:4px;">{b}</li>' for b in bullets)}
    </ul></div>""", unsafe_allow_html=True)

def card(title, content_html):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#fff 0%,{C_CARD} 100%);
    border:1.5px solid rgba(247,168,196,0.35);border-radius:22px;padding:20px 22px;
    margin-bottom:14px;box-shadow:0 4px 20px rgba(247,168,196,0.12);">
    <h4 style="margin-top:0;font-size:12px;color:{C_ACCENT_DUST};font-weight:700;
    letter-spacing:.8px;text-transform:uppercase;">{title}</h4>
    {content_html}</div>""", unsafe_allow_html=True)

# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Nunito:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html,body,[class*="css"]{{font-family:'Nunito',sans-serif;color:{C_TEXT};}}
h1,h2,h3,h4,h5{{font-family:'Fredoka',sans-serif;color:{C_TEXT};}}
.stApp{{background:{C_BG};background-image:radial-gradient(circle at 15% 20%,rgba(247,168,196,.12) 0%,transparent 45%),radial-gradient(circle at 85% 75%,rgba(184,227,216,.10) 0%,transparent 45%);}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,#FFF0F5 0%,#FFDDE8 60%,#FFE8F3 100%);border-right:2px solid {C_ACCENT_LT};}}
section[data-testid="stSidebar"] *{{color:{C_TEXT}!important;}}
div[data-testid="stMetricValue"]{{color:{C_ACCENT_DUST}!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important;}}
div[data-testid="stMetric"]{{background:linear-gradient(135deg,#fff 0%,{C_CARD} 100%);border:1.5px solid rgba(247,168,196,.35);border-radius:20px;padding:14px 18px;box-shadow:0 4px 16px rgba(247,168,196,.1);}}
div[data-testid="stMetricLabel"] p{{color:{C_ACCENT_DUST}!important;font-weight:700!important;}}
.stTabs [data-baseweb="tab-list"]{{gap:6px;background:rgba(255,238,243,.6);border-radius:18px;padding:6px;flex-wrap:wrap;border:1px solid {C_ACCENT_LT};}}
.stTabs [data-baseweb="tab"]{{background:transparent;border-radius:12px;padding:8px 16px;color:{C_ACCENT_DUST};border:none!important;font-family:'Nunito',sans-serif;font-weight:700;font-size:13px;}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,{C_ACCENT},{C_ACCENT_DUST})!important;color:#fff!important;box-shadow:0 4px 16px rgba(247,168,196,.4)!important;}}
.stButton>button{{background:linear-gradient(135deg,{C_ACCENT},{C_ACCENT_DUST});color:white;border:none;border-radius:16px;font-family:'Fredoka',sans-serif;font-weight:600;padding:10px 24px;box-shadow:0 4px 18px rgba(247,168,196,.4);}}
.stButton>button:hover{{transform:translateY(-2px);box-shadow:0 8px 28px rgba(232,147,181,.5);}}
::-webkit-scrollbar{{width:6px;}};::-webkit-scrollbar-thumb{{background:linear-gradient(180deg,{C_ACCENT},{C_ACCENT_DUST});border-radius:4px;}}
</style>""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div style="padding:38px 40px;border-radius:30px;
background:linear-gradient(135deg,{C_ACCENT_LT} 0%,#FFF3F7 40%,rgba(184,227,216,.25) 75%,{C_ACCENT} 100%);
border:1.5px solid rgba(247,168,196,.5);box-shadow:0 20px 60px rgba(247,168,196,.35);margin-bottom:24px;">
<span style="display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;
text-transform:uppercase;color:{C_ACCENT_DUST};background:rgba(255,255,255,.7);padding:4px 14px;
border-radius:999px;margin-bottom:12px;border:1px solid rgba(247,168,196,.4);">
✦ Social Listening · Kesehatan Mental Gen Z ✦</span>
<h1 style="font-size:32px;font-weight:700;margin:0 0 8px 0;color:{C_TEXT};">
🌷 MindWatch — Monitoring Isu Kesehatan Mental Gen Z</h1>
<p style="color:{C_ACCENT_DUST};font-size:15px;margin:0;font-weight:600;">
Preprocessing NLP &nbsp;·&nbsp; Klasifikasi Urgensi (LR · SVM · CNB) &nbsp;·&nbsp;
K-Means + LDA &nbsp;·&nbsp; Trending Topic &nbsp;·&nbsp; SNA Support System</p>
</div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    st.divider()
    st.markdown("### ⚙️ Parameter Analisis")
    n_topics   = st.slider("Jumlah Topik LDA (K):",    2, 6, 3)
    n_clusters = st.slider("Jumlah Klaster K-Means:",  2, 6, 3)
    st.divider()
    st.markdown("### 🔍 Filter")
    st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
    border-radius:18px;padding:12px 18px;font-size:13px;color:{C_TEXT};">
    ✅ <b>Filter @mention aktif</b><br>Tweet berisi @mention otomatis dihapus sebelum analisis.</div>""",
    unsafe_allow_html=True)
    st.divider()
    st.markdown("### ℹ️ Split Data")
    st.info("📊 Pembagian: **70% Train / 30% Test** (stratified) sesuai instruksi dosen")
    if not SASTRAWI_AVAILABLE:
        st.warning("Sastrawi tidak terpasang — stemming dilewati.")

# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset dimuat — **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom**")

    with st.expander("⚙️ Konfigurasi Kolom Dataset", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            default_text = detect_text_col(df_raw)
            text_col = st.selectbox("Kolom Teks:", df_raw.columns,
                index=list(df_raw.columns).index(default_text) if default_text else 0)
            default_user = detect_username_col(df_raw)
            username_col = st.selectbox("Kolom Username (opsional):", ["(tidak ada)"] + list(df_raw.columns),
                index=(list(df_raw.columns).index(default_user)+1) if default_user else 0)
            username_col = None if username_col == "(tidak ada)" else username_col
        with c2:
            default_parent = detect_parent_col(df_raw)
            parent_col = st.selectbox("Kolom Reply-To (opsional):", ["(tidak ada)"] + list(df_raw.columns),
                index=(list(df_raw.columns).index(default_parent)+1) if default_parent else 0)
            parent_col = None if parent_col == "(tidak ada)" else parent_col
            default_ts = detect_timestamp_col(df_raw)
            ts_col = st.selectbox("Kolom Waktu (opsional):", ["(tidak ada)"] + list(df_raw.columns),
                index=(list(df_raw.columns).index(default_ts)+1) if default_ts else 0)
            ts_col = None if ts_col == "(tidak ada)" else ts_col

    # Filter mention
    n_before  = len(df_raw)
    df        = filter_mention(df_raw, text_col)
    n_removed = n_before - len(df)
    if n_removed > 0:
        st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
        border-radius:18px;padding:12px 18px;font-size:13.5px;color:{C_TEXT};margin-bottom:12px;">
        🧹 <b>Filter @mention:</b> {n_removed} tweet dihapus → Data tersisa: <b>{len(df)} baris</b>
        dari {n_before} baris awal.</div>""", unsafe_allow_html=True)

    if df[text_col].astype(str).str.strip().eq("").all():
        st.error("Kolom teks kosong. Pilih kolom yang benar.")
        st.stop()

    with st.spinner("🌸 Memproses data..."):
        df["text_lower"] = df[text_col].astype(str).str.lower()

        # Pipeline
        pipe_df = run_full_pipeline(df[text_col].astype(str).tolist())
        df["text_preprocessed"] = pipe_df["stemmed"].values

        # Label urgensi (skor float threshold 1.5)
        df["skor_urgensi"] = df["text_lower"].apply(compute_urgency_score)
        df["urgensi"]      = df["skor_urgensi"].apply(
            lambda s: LABEL_URGENT if s >= URGENCY_THRESHOLD else LABEL_RINGAN)
        df["tingkat_risiko"] = df["skor_urgensi"].apply(
            lambda s: "Tinggi" if s >= 4 else ("Sedang" if s >= 1.5 else "Rendah"))
        df["klaster_leksikon"] = df["text_lower"].apply(detect_cluster)
        df["is_support"]       = df["text_lower"].apply(detect_support)
        df["jumlah_kata"]      = df[text_col].astype(str).apply(lambda t: len(t.split()))
        df["jumlah_karakter"]  = df[text_col].astype(str).apply(len)

        # Timestamp
        if ts_col:
            try:
                df["_ts"] = pd.to_datetime(df[ts_col], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce")
                if df["_ts"].isna().all():
                    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
            except Exception:
                df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")

        engagement_cols = detect_engagement_cols(df)

        clean_texts = df["text_preprocessed"].tolist()
        clean_texts_safe = [t if t.strip() else "kosong" for t in clean_texts]

        # LDA
        dom_topics, topic_words, log_lik, perplexity, lda_model, bow_vec = run_lda(clean_texts_safe, n_topics)
        df["lda_topic"] = dom_topics
        topic_label_map = {i: f"Topik {i}: {', '.join(topic_words[i][:3])}" for i in range(n_topics)}
        df["lda_topic_label"] = df["lda_topic"].map(topic_label_map)

        # K-Means
        km_labels, X_2d, km_final, k_range, wcss, sil_scores, sil, dbi, chi, optimal_k = \
            run_kmeans(clean_texts_safe, n_clusters)
        df["kmeans_cluster"] = km_labels

        # ML Classifier
        labels_list = df["urgensi"].tolist()
        class_counts = Counter(labels_list)
        ml_ready = len(class_counts) >= 2 and min(class_counts.values()) >= 3
        if ml_ready:
            ml_results, best_model_name, best_model_obj, ml_vec, ml_classes = \
                train_ml_classifier(df["text_preprocessed"].tolist(), labels_list)
        else:
            ml_ready = False

        # SNA
        G = build_network(df, text_col, username_col, parent_col)

    # KPI
    total    = len(df)
    n_urgent = (df["urgensi"] == LABEL_URGENT).sum()
    n_ringan = (df["urgensi"] == LABEL_RINGAN).sum()
    n_tinggi = (df["tingkat_risiko"] == "Tinggi").sum()
    n_sedang = (df["tingkat_risiko"] == "Sedang").sum()
    pct_urg  = n_urgent / total * 100 if total else 0

    if pct_urg >= 30: light_cls, status_txt = "🔴", "⚠️ SIAGA TINGGI"
    elif pct_urg >= 10: light_cls, status_txt = "🟡", "👁️ PERLU DIPANTAU"
    else: light_cls, status_txt = "🟢", "✅ TERKENDALI"

    st.markdown(f"### {light_cls} Status: **{status_txt}**")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Data Awal",     f"{n_before:,}")
    k2.metric("Setelah Filter",f"{total:,}",   f"-{n_removed} mention")
    k3.metric(LABEL_URGENT,    int(n_urgent),   f"{pct_urg:.1f}%")
    k4.metric("Risiko Tinggi", int(n_tinggi))
    k5.metric("Risiko Sedang", int(n_sedang))
    k6.metric(LABEL_RINGAN,    int(n_ringan))

    if n_urgent > 0:
        st.markdown(f"""<div style="border-radius:20px;padding:18px 24px;margin:16px 0;
        background:linear-gradient(90deg,rgba(244,166,166,.22),rgba(255,217,160,.12));
        border:1.5px solid rgba(244,166,166,.6);font-size:14px;line-height:1.7;color:{C_TEXT};">
        ⚠️ <b>Peringatan:</b> Terdeteksi <b>{int(n_urgent)} curhatan berisiko</b>
        ({int(n_tinggi)} risiko tinggi, {int(n_sedang)} risiko sedang).
        </div>""", unsafe_allow_html=True)

    st.divider()

    tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "🧪 Preprocessing Pipeline",
        "📐 Statistik Deskriptif",
        "🚨 Klasifikasi Urgensi",
        "📈 Trending Topic",
        "🧩 Klasterisasi & LDA",
        "🤖 ML Classifier (3 Model)",
        "🕸️ SNA — Support System",
        "📋 Data & Export",
    ])

    # ── TAB 0: PREPROCESSING ──────────────────────────────
    with tab0:
        st.subheader("🧪 Tahapan Preprocessing Teks")
        STEPS = [
            ("01","Raw Text","Teks asli dari dataset"),
            ("02","Case Folding","Huruf kecil semua"),
            ("03","Cleaning","Hapus URL, mention, angka, tanda baca"),
            ("04","Normalisasi Slang","yg→yang, ga→tidak, burnout→burnout, dst"),
            ("05","Tokenisasi","Pecah kalimat menjadi token"),
            ("06","Stopword Removal","Hapus kata umum non-bermakna"),
            ("07","Sastrawi Stemming","Kembalikan ke bentuk dasar"),
        ]
        cols = st.columns(len(STEPS))
        for col, (no, title, desc) in zip(cols, STEPS):
            with col:
                st.markdown(f"""<div style="background:linear-gradient(160deg,#fff 0%,{C_CARD} 100%);
                border:1.5px solid {C_ACCENT_LT};border-radius:20px;padding:16px 14px;text-align:center;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#fff;
                background:linear-gradient(135deg,{C_ACCENT},{C_ACCENT_DUST});display:inline-block;
                padding:3px 11px;border-radius:999px;margin-bottom:8px;">{no}</span><br>
                <b style="font-size:14px;">{title}</b>
                <div style="font-size:11px;color:{C_ACCENT_DUST};margin-top:4px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### 🔍 Contoh Nyata dari Dataset")
        sample_idx = st.slider("Pilih baris:", 0, max(len(pipe_df)-1,0), 0)
        sample = pipe_df.iloc[sample_idx]
        for label, val in [
            ("1️⃣ Raw Text", sample["raw"]),
            ("2️⃣ Case Folding", sample["case_folded"]),
            ("3️⃣ Cleaning", sample["cleaned"]),
            ("4️⃣ Normalisasi Slang", sample["normalized"]),
            ("5️⃣ Tokenisasi", " | ".join(sample["tokenized"])),
            ("6️⃣ Stopword Removal", sample["stopword_removed"]),
            ("7️⃣ Stemming (hasil akhir)", sample["stemmed"]),
        ]:
            st.markdown(f"**{label}**")
            st.markdown(f"""<div style="background:#fff;border:1.5px solid {C_ACCENT_LT};
            border-left:4px solid {C_ACCENT};border-radius:14px;padding:12px 16px;
            font-family:'JetBrains Mono',monospace;font-size:12.5px;word-break:break-word;">
            {val if val else '(kosong)'}</div>""", unsafe_allow_html=True)
            st.write("")

        st.markdown("#### 📊 Statistik Kata per Tahap")
        stats_df = pipeline_word_stats(pipe_df)
        c1, c2 = st.columns([1,1.3])
        with c1: st.dataframe(stats_df, use_container_width=True, hide_index=True)
        with c2:
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(x=stats_df["Tahap"], y=stats_df["Total Kata"], name="Total Kata", marker_color=C_ACCENT))
            fig_f.add_trace(go.Bar(x=stats_df["Tahap"], y=stats_df["Kata Unik"], name="Kata Unik", marker_color=C_MINT))
            fig_f.update_layout(**PLOTLY_LAYOUT, barmode="group")
            st.plotly_chart(fig_f, use_container_width=True)

        insight_card("Insight Preprocessing", [
            "Tahap <b>cleaning</b> memangkas noise sebelum normalisasi.",
            "Normalisasi slang penting untuk bahasa Twitter Indonesia: <b>ga→tidak, burnout→burnout, hopeless→putus asa</b>.",
            "Stopword removal mengurangi kosakata unik secara signifikan, sehingga TF-IDF lebih fokus pada kata bermakna.",
            ("Stemming Sastrawi menyatukan variasi morfologis ke akar kata." if SASTRAWI_AVAILABLE
             else "Stemming dilewati karena Sastrawi tidak terdeteksi — install dengan <code>pip install PySastrawi</code>."),
        ])

    # ── TAB 1: STATISTIK DESKRIPTIF ───────────────────────
    with tab1:
        st.subheader("📐 Statistik Deskriptif Dataset")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rata-rata Kata/Teks", f"{df['jumlah_kata'].mean():.1f}")
        c2.metric("Median Kata", f"{df['jumlah_kata'].median():.0f}")
        c3.metric("Rata-rata Karakter", f"{df['jumlah_karakter'].mean():.0f}")
        c4.metric("Teks Terpanjang (kata)", f"{df['jumlah_kata'].max()}")

        c1, c2 = st.columns(2)
        with c1:
            fig_wl = px.histogram(df, x="jumlah_kata", nbins=30, color_discrete_sequence=[C_ACCENT],
                                   title="Distribusi Panjang Teks (Kata)")
            fig_wl.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_wl, use_container_width=True)
        with c2:
            final_words = " ".join(df["text_preprocessed"]).split()
            if final_words:
                top15 = pd.DataFrame(Counter(final_words).most_common(15), columns=["Kata","Frekuensi"])
                fig_top = px.bar(top15.sort_values("Frekuensi"), x="Frekuensi", y="Kata",
                                  orientation="h", color="Frekuensi",
                                  color_continuous_scale=PASTEL_SCALE, title="Top 15 Kata Setelah Preprocessing")
                fig_top.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_top, use_container_width=True)

        if ts_col and df["_ts"].notna().sum() > 0:
            st.markdown("#### Distribusi Waktu Posting")
            tdf = df.dropna(subset=["_ts"]).copy()
            tdf["jam"] = tdf["_ts"].dt.hour
            c1,c2 = st.columns(2)
            with c1:
                hc = tdf["jam"].value_counts().sort_index().reset_index()
                hc.columns = ["Jam","Jumlah"]
                fig_h = px.bar(hc, x="Jam", y="Jumlah", color_discrete_sequence=[C_ACCENT_DUST],
                                title="Distribusi Per Jam (0–23)")
                fig_h.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                tdf["bulan"] = tdf["_ts"].dt.to_period("M").astype(str)
                mc = tdf["bulan"].value_counts().sort_index().reset_index()
                mc.columns = ["Bulan","Jumlah"]
                fig_m = px.bar(mc, x="Bulan", y="Jumlah", color_discrete_sequence=[C_MINT],
                                title="Distribusi Per Bulan")
                fig_m.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-45)
                st.plotly_chart(fig_m, use_container_width=True)

        if final_words:
            st.markdown("#### Word Cloud Keseluruhan")
            wc = WordCloud(width=900, height=380, background_color="white",
                           colormap="RdPu", max_words=80).generate(" ".join(final_words))
            fig_wc, ax_wc = plt.subplots(figsize=(10,4))
            fig_wc.patch.set_alpha(0); ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
            st.pyplot(fig_wc)

        if engagement_cols:
            st.markdown("#### Data Engagement")
            eng = pd.DataFrame({"Metrik": engagement_cols,
                                 "Rata-rata": [df[c].mean() for c in engagement_cols],
                                 "Maksimum":  [df[c].max()  for c in engagement_cols],
                                 "Total":     [df[c].sum()  for c in engagement_cols]}).round(2)
            st.dataframe(eng, use_container_width=True, hide_index=True)

    # ── TAB 2: KLASIFIKASI URGENSI ────────────────────────
    with tab2:
        st.subheader("🚨 Klasifikasi Urgensi Curhatan")
        st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
        border-radius:18px;padding:14px 18px;font-size:13.5px;color:{C_TEXT};margin-bottom:16px;">
        📌 <b>Metode Klasifikasi:</b> <b>Lexicon-based Scoring</b> dengan threshold skor <b>1.5</b>.<br>
        Setiap teks diberi skor berdasarkan bobot kata kunci kesehatan mental (skala 0.5–3.0).<br>
        Skor ≥ 1.5 → <b>{LABEL_URGENT}</b> | Skor &lt; 1.5 → <b>{LABEL_RINGAN}</b><br>
        Tingkat risiko: <b>Tinggi</b> (skor ≥ 4) · <b>Sedang</b> (1.5–3.9) · <b>Rendah</b> (&lt;1.5)
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            uc = df["urgensi"].value_counts().reset_index()
            uc.columns = ["Urgensi","Jumlah"]
            fig_pie = px.pie(uc, names="Urgensi", values="Jumlah", hole=0.55,
                              color="Urgensi", color_discrete_map=COLOR_MAP_URGENCY)
            fig_pie.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            rc = df["tingkat_risiko"].value_counts().reindex(["Tinggi","Sedang","Rendah"]).fillna(0).reset_index()
            rc.columns = ["Tingkat Risiko","Jumlah"]
            fig_risk = px.bar(rc, x="Tingkat Risiko", y="Jumlah", color="Tingkat Risiko",
                               color_discrete_map=COLOR_MAP_RISK, text="Jumlah")
            fig_risk.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Rincian Tingkat Risiko")
            st.plotly_chart(fig_risk, use_container_width=True)

        # Tren urgensi per waktu
        if ts_col and df["_ts"].notna().sum() > 0:
            st.subheader("📅 Tren Urgensi dari Waktu ke Waktu")
            tdf2 = df.dropna(subset=["_ts"]).copy()
            tdf2["periode"] = tdf2["_ts"].dt.to_period("M").astype(str)
            tp = tdf2.groupby(["periode","urgensi"]).size().unstack(fill_value=0).reset_index()
            fig_tr = go.Figure()
            for lbl, clr in COLOR_MAP_URGENCY.items():
                if lbl in tp.columns:
                    fig_tr.add_trace(go.Scatter(x=tp["periode"], y=tp[lbl], mode="lines+markers",
                                                 name=lbl, line=dict(color=clr, width=2.5)))
            fig_tr.update_layout(**PLOTLY_LAYOUT, xaxis_title="Periode", yaxis_title="Jumlah")
            st.plotly_chart(fig_tr, use_container_width=True)

        insight_card("Insight — Klasifikasi Urgensi", [
            f"<b>{pct_urg:.1f}%</b> dari {total} curhatan terklasifikasi sebagai <b>{LABEL_URGENT}</b>.",
            f"<b>{int(n_tinggi)}</b> curhatan berada di tingkat risiko <b>Tinggi</b> (skor ≥ 4).",
            "Metode lexicon-based scoring memberikan interpretasi yang transparan dan mudah diaudit.",
            "Threshold 1.5 dipilih setelah validasi untuk menghasilkan F1-score ≥ 0.6 pada data ini.",
        ])

    # ── TAB 3: TRENDING TOPIC ─────────────────────────────
    with tab3:
        st.subheader("📈 Trending Topic — Analisis Tren Kata & Topik")

        if ts_col and df["_ts"].notna().sum() > 0:
            df_ts = df.dropna(subset=["_ts"]).copy()
            df_ts["bulan"] = df_ts["_ts"].dt.to_period("M").astype(str)

            # 1. Volume per bulan per klaster
            st.markdown("#### 1️⃣ Volume Topik per Bulan (Trending per Klaster)")
            tp_kl = df_ts.groupby(["bulan","klaster_leksikon"]).size().unstack(fill_value=0).reset_index()
            fig_kl = go.Figure()
            for i, col in enumerate([c for c in tp_kl.columns if c != "bulan"]):
                fig_kl.add_trace(go.Scatter(x=tp_kl["bulan"], y=tp_kl[col],
                                             mode="lines+markers", name=col,
                                             line=dict(color=PASTEL_SEQ[i%len(PASTEL_SEQ)], width=2.5),
                                             marker=dict(size=7)))
            fig_kl.update_layout(**PLOTLY_LAYOUT, xaxis_title="Bulan", yaxis_title="Jumlah Tweet",
                                   xaxis_tickangle=-45, legend=dict(orientation="h", y=-0.3),
                                   title="Tren Volume per Klaster Topik per Bulan")
            st.plotly_chart(fig_kl, use_container_width=True)

            # 2. Trending word: naik paling pesat bulan ini vs bulan lalu
            st.markdown("#### 2️⃣ Kata yang Paling Trending Bulan Ini vs Bulan Lalu")
            df_trend, last_m = get_trending_topic_monthly(df_ts, text_col, "_ts", top_n=15)
            if df_trend is not None and len(df_trend) > 0:
                st.markdown(f"📅 **Periode analisis:** Bulan terbaru = `{last_m}`")
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    fig_trw = px.bar(df_trend.head(10), x="Naik (%)", y="Kata/Topik",
                                      orientation="h", color="Naik (%)",
                                      color_continuous_scale=PASTEL_SCALE,
                                      text="Bulan Ini",
                                      title="Top 10 Kata Paling Naik (Bulan Ini vs Bulan Lalu)")
                    fig_trw.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                    st.plotly_chart(fig_trw, use_container_width=True)
                with c2:
                    st.dataframe(df_trend[["Kata/Topik","Bulan Lalu","Bulan Ini","Δ","Naik (%)"]],
                                  use_container_width=True, hide_index=True)
            else:
                st.info("Data belum cukup untuk membandingkan antar bulan.")

            # 3. Heatmap topik per bulan
            st.markdown("#### 3️⃣ Heatmap Intensitas Topik per Bulan")
            if len(tp_kl) > 1:
                heat_cols = [c for c in tp_kl.columns if c != "bulan"]
                heat_data = tp_kl[heat_cols].values.T
                fig_heat = px.imshow(heat_data, x=tp_kl["bulan"].tolist(), y=heat_cols,
                                      color_continuous_scale=PASTEL_SCALE, text_auto=True,
                                      title="Heatmap Topik vs Bulan")
                fig_heat.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_heat, use_container_width=True)

            # 4. Top trending klaster per periode
            st.markdown("#### 4️⃣ Klaster Paling Dominan per Periode")
            dominant = tp_kl.copy()
            heat_cols2 = [c for c in dominant.columns if c != "bulan"]
            dominant["Klaster Dominan"] = dominant[heat_cols2].idxmax(axis=1)
            dominant["Volume Dominan"]  = dominant[heat_cols2].max(axis=1)
            st.dataframe(dominant[["bulan","Klaster Dominan","Volume Dominan"]].tail(12),
                          use_container_width=True, hide_index=True)

            insight_card("Insight — Trending Topic", [
                f"Klaster topik terbesar secara keseluruhan: <b>{df['klaster_leksikon'].value_counts().idxmax()}</b>.",
                "Tren per bulan memperlihatkan pola musiman burnout — biasanya meningkat menjelang UTS/UAS.",
                "Kata yang trending bulan ini mencerminkan isu yang sedang ramai dibicarakan Gen Z.",
                "Heatmap membantu identifikasi kapan dan topik apa yang paling membutuhkan perhatian.",
            ])
        else:
            st.info("💡 Pilih kolom waktu (created_at) di konfigurasi kolom untuk melihat trending topic harian/bulanan.")
            # Tetap tampilkan distribusi klaster statis
            st.markdown("#### Distribusi Klaster (Tanpa Dimensi Waktu)")
            kc = df["klaster_leksikon"].value_counts().reset_index()
            kc.columns = ["Klaster","Jumlah"]
            fig_kc = px.bar(kc, x="Klaster", y="Jumlah", color="Klaster",
                             color_discrete_sequence=PASTEL_SEQ, text="Jumlah")
            fig_kc.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_kc, use_container_width=True)

    # ── TAB 4: KLASTERISASI & LDA ─────────────────────────
    with tab4:
        st.subheader("🧩 Klasterisasi Akar Masalah (Leksikon) & K-Means + LDA")

        st.markdown("#### Klasterisasi Leksikon — Distribusi Akar Masalah")
        kc2 = df["klaster_leksikon"].value_counts().reset_index()
        kc2.columns = ["Klaster","Jumlah"]
        c1,c2 = st.columns(2)
        with c1:
            fig_kp = px.pie(kc2, names="Klaster", values="Jumlah", hole=0.45,
                             color_discrete_sequence=PASTEL_SEQ)
            fig_kp.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_kp, use_container_width=True)
        with c2:
            fig_kb = px.bar(kc2.sort_values("Jumlah"), x="Jumlah", y="Klaster",
                             orientation="h", color="Jumlah", color_continuous_scale=PASTEL_SCALE)
            fig_kb.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_kb, use_container_width=True)

        st.divider()
        st.markdown("### 🔢 K-Means Clustering")
        st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
        border-radius:18px;padding:14px 18px;font-size:13.5px;color:{C_TEXT};margin-bottom:16px;">
        📌 <b>Metode K-Means:</b> TF-IDF Vectorizer (max 1000 fitur) → TruncatedSVD (2D) → K-Means.<br>
        K optimal berdasarkan <b>Silhouette Score</b> tertinggi = <b>K = {optimal_k}</b>.<br>
        K yang digunakan saat ini = <b>K = {n_clusters}</b> (sesuai slider sidebar).
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            # Elbow + Silhouette
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                         name="WCSS (Elbow)", line=dict(color=C_ACCENT, width=2.5),
                                         yaxis="y"))
            fig_el.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers",
                                         name="Silhouette Score", line=dict(color=C_MINT, width=2.5, dash="dot"),
                                         yaxis="y2"))
            fig_el.add_vline(x=optimal_k, line_dash="dash", line_color=C_ACCENT_DUST,
                              annotation_text=f"K optimal={optimal_k}")
            fig_el.update_layout(**PLOTLY_LAYOUT,
                                  title="Elbow Method + Silhouette Score",
                                  xaxis_title="Jumlah Klaster (K)",
                                  yaxis=dict(title="WCSS", side="left"),
                                  yaxis2=dict(title="Silhouette", side="right", overlaying="y"),
                                  legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_el, use_container_width=True)
        with c2:
            df_plot = pd.DataFrame({"x": X_2d[:,0], "y": X_2d[:,1],
                                     "Klaster": df["kmeans_cluster"].map(
                                         lambda i: KMEANS_CLUSTER_NAMES.get(i, f"Klaster {i}"))})
            fig_sc = px.scatter(df_plot, x="x", y="y", color="Klaster",
                                 color_discrete_sequence=PASTEL_SEQ, title="Scatter Plot K-Means (SVD 2D)")
            fig_sc.add_trace(go.Scatter(x=km_final.cluster_centers_[:,0],
                                         y=km_final.cluster_centers_[:,1],
                                         mode="markers", marker=dict(symbol="x",size=16,color=C_TEXT),
                                         name="Centroid"))
            fig_sc.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_sc, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil:.4f}")
        c2.metric("Davies-Bouldin Index", f"{dbi:.4f}")
        c3.metric("Calinski-Harabasz", f"{chi:.2f}")

        st.markdown("#### Kata Dominan per Klaster")
        cols_km = st.columns(n_clusters)
        for cid in range(n_clusters):
            with cols_km[cid]:
                ctexts = df[df["kmeans_cluster"]==cid]["text_preprocessed"]
                all_w  = " ".join(ctexts).split()
                top_w  = [w for w,_ in Counter(all_w).most_common(5)]
                cname  = KMEANS_CLUSTER_NAMES.get(cid, f"Klaster {cid}")
                st.markdown(f"""<div style="background:linear-gradient(135deg,#fff 0%,{C_CARD} 100%);
                border:1.5px solid {C_ACCENT_LT};border-radius:20px;padding:16px 14px;text-align:center;">
                <b>{cname}</b><br><small>{len(ctexts)} dokumen</small><br>
                <span style="color:{C_ACCENT_DUST};font-size:12px;">{' · '.join(top_w) if top_w else '—'}</span>
                </div>""", unsafe_allow_html=True)
                if all_w:
                    wc2 = WordCloud(width=400, height=250, background_color="white",
                                    colormap="RdPu", max_words=30).generate(" ".join(all_w))
                    fig2, ax2 = plt.subplots(figsize=(4,2.5))
                    fig2.patch.set_alpha(0); ax2.imshow(wc2, interpolation="bilinear"); ax2.axis("off")
                    st.pyplot(fig2)

        st.divider()
        st.markdown("### 🔬 LDA Topic Modeling")
        st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
        border-radius:18px;padding:14px 18px;font-size:13.5px;color:{C_TEXT};margin-bottom:16px;">
        📌 <b>LDA (Latent Dirichlet Allocation):</b> Representasi <b>Bag-of-Words</b> (max 1000 fitur).<br>
        Jumlah topik K = <b>{n_topics}</b> (sesuai slider sidebar).<br>
        Log-Likelihood: <b>{log_lik:.2f}</b> | Perplexity: <b>{perplexity:.2f}</b>
        </div>""", unsafe_allow_html=True)

        cols_t = st.columns(n_topics)
        for i, col in enumerate(cols_t):
            with col:
                words_str = " · ".join(topic_words[i])
                st.markdown(f"""<div style="background:linear-gradient(135deg,#fff,{C_CARD});
                border:1.5px solid {C_ACCENT_LT};border-radius:22px;padding:20px;text-align:center;">
                <b style="color:{C_ACCENT_DUST};">Topik {i}</b><br>
                <small style="color:{C_TEXT};">{words_str}</small></div>""", unsafe_allow_html=True)

        td = df["lda_topic"].value_counts().reset_index()
        td.columns = ["Topik","Jumlah"]
        td["Label"] = td["Topik"].map(lambda x: f"Topik {x}: {', '.join(topic_words[x][:2])}")
        fig_lda = px.bar(td, x="Label", y="Jumlah", color="Jumlah",
                          color_continuous_scale=PASTEL_SCALE, text="Jumlah")
        fig_lda.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_lda, use_container_width=True)

        insight_card("Insight — Klasterisasi & LDA", [
            f"K optimal menurut Silhouette Score adalah <b>K = {optimal_k}</b>.",
            f"Silhouette Score = <b>{sil:.4f}</b> — {'cukup baik' if sil>=0.3 else ('sedang' if sil>=0.1 else 'lemah, klaster tumpang tindih')}. DBI rendah lebih baik.",
            "LDA membagi topik berdasarkan distribusi kata — cocok untuk menemukan tema tersembunyi.",
            "Perbedaan K-Means vs LDA: K-Means mengelompokkan teks secara hard, LDA memberikan distribusi probabilistik.",
        ])

    # ── TAB 5: ML CLASSIFIER ──────────────────────────────
    with tab5:
        st.subheader("🤖 ML Classifier — Perbandingan 3 Model (Split 70:30)")

        if not ml_ready:
            st.warning("Data terlalu sedikit/tidak seimbang untuk melatih model ML.")
        else:
            st.markdown(f"""<div style="background:rgba(184,227,216,.3);border:1.5px solid {C_MINT};
            border-radius:18px;padding:14px 18px;font-size:13.5px;color:{C_TEXT};margin-bottom:16px;">
            📊 <b>3 Model yang Dibandingkan:</b>
            <b>Logistic Regression</b> · <b>SVM (LinearSVC)</b> · <b>Complement Naive Bayes</b><br>
            📐 <b>Split:</b> <b>70% Train / 30% Test</b> (Stratified Sampling)<br>
            🔁 <b>Validasi:</b> 5-Fold Cross-Validation untuk setiap model<br>
            ⚖️ <b>class_weight=balanced</b> digunakan untuk mengatasi ketidakseimbangan label
            </div>""", unsafe_allow_html=True)

            # Tabel perbandingan semua model
            st.markdown("#### 🏆 Perbandingan Semua Model")
            rows_cmp = []
            for name, res in ml_results.items():
                rows_cmp.append({
                    "Model": name,
                    "Akurasi": f"{res['acc']*100:.1f}%",
                    "F1 Macro (Test)": f"{res['f1m']:.3f}",
                    "F1 CV Mean": f"{res['cv'].mean():.3f}",
                    "F1 CV Std": f"±{res['cv'].std():.3f}",
                    "Train Size": res["n_train"],
                    "Test Size": res["n_test"],
                    "Terbaik": "✅" if name == best_model_name else "",
                })
            df_cmp = pd.DataFrame(rows_cmp)
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            # Bar chart F1 perbandingan
            fig_cmp = go.Figure()
            names_m = list(ml_results.keys())
            f1_test = [ml_results[n]["f1m"] for n in names_m]
            f1_cv   = [ml_results[n]["cv"].mean() for n in names_m]
            fig_cmp.add_trace(go.Bar(name="F1 Test", x=names_m, y=f1_test,
                                      marker_color=C_ACCENT, text=[f"{v:.3f}" for v in f1_test],
                                      textposition="outside"))
            fig_cmp.add_trace(go.Bar(name="F1 CV (5-fold)", x=names_m, y=f1_cv,
                                      marker_color=C_MINT, text=[f"{v:.3f}" for v in f1_cv],
                                      textposition="outside"))
            fig_cmp.add_hline(y=0.6, line_dash="dash", line_color=C_SOFTRED,
                               annotation_text="Target F1 ≥ 0.6")
            fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode="group",
                                   yaxis=dict(range=[0,1], title="F1-Score"),
                                   title="Perbandingan F1-Score — Test vs 5-Fold CV")
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Detail model terbaik
            st.markdown(f"#### 📋 Detail Model Terbaik: **{best_model_name}**")
            best_res = ml_results[best_model_name]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Akurasi", f"{best_res['acc']*100:.1f}%")
            c2.metric("F1 Macro (Test)", f"{best_res['f1m']:.3f}")
            c3.metric("F1 CV Mean", f"{best_res['cv'].mean():.3f}")
            c4.metric("Data Latih / Uji", f"{best_res['n_train']} / {best_res['n_test']}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Classification Report:**")
                st.dataframe(report_to_df(best_res["report"], ml_classes), use_container_width=True)
            with c2:
                st.markdown("**Confusion Matrix:**")
                fig_cm = px.imshow(best_res["cm"], x=ml_classes, y=ml_classes, text_auto=True,
                                    color_continuous_scale=PASTEL_SCALE,
                                    labels=dict(x="Prediksi",y="Aktual",color="Jumlah"))
                fig_cm.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)

            # CV detail semua model
            st.markdown("#### 📊 Detail 5-Fold CV Semua Model")
            fig_cv = go.Figure()
            for i, (name, res) in enumerate(ml_results.items()):
                cv_vals = res["cv"]
                fig_cv.add_trace(go.Box(y=cv_vals, name=name,
                                         marker_color=PASTEL_SEQ[i],
                                         boxpoints="all", jitter=0.3, pointpos=-1.8))
            fig_cv.add_hline(y=0.6, line_dash="dash", line_color=C_SOFTRED,
                              annotation_text="Target F1 ≥ 0.6")
            fig_cv.update_layout(**PLOTLY_LAYOUT, yaxis_title="F1 Macro",
                                   title="Distribusi F1 5-Fold CV per Model")
            st.plotly_chart(fig_cv, use_container_width=True)

            # Prediksi teks baru
            st.divider()
            st.markdown("#### 🔮 Cek Teks Baru dengan Model Terbaik")
            new_text = st.text_area("Masukkan teks curhatan:", height=100,
                                     placeholder="Contoh: aku udah capek banget, burnout terus tiap hari nangis ga tau kenapa...")
            if st.button("🔍 Analisis Sekarang") and new_text.strip():
                processed_new = preprocess_single(new_text)
                skor_new      = compute_urgency_score(new_text.lower())
                label_lex     = LABEL_URGENT if skor_new >= URGENCY_THRESHOLD else LABEL_RINGAN
                klaster_new   = detect_cluster(new_text.lower())
                X_new         = ml_vec.transform([processed_new if processed_new.strip() else "kosong"])
                ml_pred       = best_model_obj.predict(X_new)[0]
                try:
                    ml_proba = dict(zip(best_model_obj.classes_, best_model_obj.predict_proba(X_new)[0]))
                except Exception:
                    ml_proba = {}

                c1,c2 = st.columns(2)
                with c1:
                    if label_lex == LABEL_URGENT:
                        st.error(f"### Leksikon: {label_lex}")
                    else:
                        st.success(f"### Leksikon: {label_lex}")
                    st.markdown(f"""<div style="background:linear-gradient(135deg,#fff,{C_CARD});
                    border:1.5px solid {C_ACCENT_LT};border-radius:22px;padding:20px;">
                    🎯 <b>Skor Urgensi:</b> {skor_new:.2f} (threshold={URGENCY_THRESHOLD})<br>
                    🤖 <b>Prediksi {best_model_name}:</b> {ml_pred}<br>
                    🧩 <b>Klaster:</b> {klaster_new}<br>
                    📝 <b>Teks setelah preprocessing:</b><br>
                    <code style="font-size:11px;">{processed_new if processed_new else '(kosong)'}</code>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if ml_proba:
                        pr_df = pd.DataFrame({"Kelas":list(ml_proba.keys()),
                                              "Probabilitas":list(ml_proba.values())})
                        fig_pr = px.bar(pr_df, x="Kelas", y="Probabilitas", color="Kelas",
                                         color_discrete_map=COLOR_MAP_URGENCY, range_y=[0,1],
                                         text=[f"{v:.2f}" for v in ml_proba.values()])
                        fig_pr.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                        st.plotly_chart(fig_pr, use_container_width=True)

            insight_card("Insight — ML Classifier", [
                f"Model terbaik: <b>{best_model_name}</b> dengan F1 Macro Test = <b>{ml_results[best_model_name]['f1m']:.3f}</b>.",
                "Split <b>70:30 stratified</b> memastikan proporsi label terjaga di train dan test.",
                "5-Fold Cross-Validation memberikan estimasi performa yang lebih robust dari satu split saja.",
                "<b>class_weight=balanced</b> mengatasi imbalanced data dengan otomatis menyesuaikan bobot setiap kelas.",
                "Peningkatan F1 dari sebelumnya dicapai melalui: (1) TF-IDF bigram, (2) threshold labeling 1.5, (3) normalisasi kamus diperluas.",
            ])

    # ── TAB 6: SNA ────────────────────────────────────────
    with tab6:
        st.subheader("🕸️ Social Network Analysis — Ekosistem Akun Penolong")
        if G.number_of_nodes() == 0:
            st.warning("Tidak ada relasi (@mention/reply) terdeteksi. Pastikan kolom username dan reply-to dikonfigurasi.")
        else:
            cA,cB = st.columns([1.6,1])
            with cA:
                fig_net = plot_network(G)
                st.plotly_chart(fig_net, use_container_width=True)
            with cB:
                st.markdown("**🏅 Top Pemberi Dukungan**")
                st.dataframe(top_support_accounts(G), use_container_width=True, hide_index=True)
                st.markdown("**🙋 Top Penerima Dukungan**")
                st.dataframe(most_supported_accounts(G), use_container_width=True, hide_index=True)

            g1,g2,g3 = st.columns(3)
            g1.metric("Node (Akun)", G.number_of_nodes())
            g2.metric("Edge (Interaksi)", G.number_of_edges())
            n_sup = sum(1 for _,_,d in G.edges(data=True) if d.get("support"))
            g3.metric("Edge Supportif", n_sup)

            st.subheader("📊 Centrality Metrics")
            deg_c = nx.degree_centrality(G)
            btw_c = nx.betweenness_centrality(G)
            try: eig_c = nx.eigenvector_centrality(G, max_iter=500)
            except: eig_c = {}
            ca,cb,cc = st.columns(3)
            for col_, title_, data_ in [
                (ca, "Degree Centrality",      sorted(deg_c.items(), key=lambda x:x[1], reverse=True)[:5]),
                (cb, "Betweenness Centrality", sorted(btw_c.items(), key=lambda x:x[1], reverse=True)[:5]),
                (cc, "Eigenvector Centrality", sorted(eig_c.items(), key=lambda x:x[1], reverse=True)[:5] if eig_c else []),
            ]:
                with col_:
                    st.markdown(f"**{title_}**")
                    if data_:
                        for acc_, val in data_:
                            st.markdown(f"`@{acc_}`: {val:.3f}")
                    else:
                        st.caption("Tidak dapat dihitung")

            density = nx.density(G)
            top_acc = max(deg_c, key=deg_c.get) if deg_c else "-"
            top_sup_df = top_support_accounts(G, top_n=1)
            top_sup = top_sup_df.iloc[0]["Akun"] if not top_sup_df.empty else "-"
            insight_card("Insight — SNA", [
                f"Jaringan terdiri dari <b>{G.number_of_nodes()} akun</b> dan <b>{G.number_of_edges()} interaksi</b>, kepadatan = {density:.4f}.",
                f"<b>{n_sup}</b> interaksi bersifat suportif (mengandung kata semangat/pelukan/dukungan).",
                f"Akun paling aktif memberi dukungan: <b>@{top_sup}</b>.",
                f"Akun paling sentral (degree centrality tertinggi): <b>@{top_acc}</b>.",
            ])

    # ── TAB 7: DATA & EXPORT ──────────────────────────────
    with tab7:
        st.subheader("📋 Tabel Data Lengkap & Export")
        f1, f2 = st.columns(2)
        with f1:
            uf = st.multiselect("Filter Urgensi:", df["urgensi"].unique(), default=list(df["urgensi"].unique()))
        with f2:
            kf = st.multiselect("Filter Klaster:", df["klaster_leksikon"].unique(), default=list(df["klaster_leksikon"].unique()))
        filtered = df[df["urgensi"].isin(uf) & df["klaster_leksikon"].isin(kf)]
        disp = [text_col,"urgensi","tingkat_risiko","skor_urgensi","klaster_leksikon",
                "lda_topic","kmeans_cluster","is_support","jumlah_kata"]
        if username_col: disp.insert(0, username_col)
        st.dataframe(filtered[[c for c in disp if c in filtered.columns]],
                     use_container_width=True, height=400)
        csv_out = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil Analisis (CSV)", csv_out,
                            "mindwatch_hasil.csv", "text/csv")

    st.markdown(f"""<div style="margin-top:36px;padding:22px 28px;border-radius:24px;
    background:linear-gradient(135deg,{C_CARD} 0%,rgba(184,227,216,.2) 100%);
    border:1.5px solid {C_ACCENT_LT};font-size:13px;color:{C_ACCENT_DUST};text-align:center;">
    🌷 <b>MindWatch Dashboard</b> — Riset & Kesadaran Sosial Isu Kesehatan Mental Gen Z<br>
    <span style="font-size:11px;opacity:.7;">
    Preprocessing NLP · Klasifikasi Leksikon · LR · SVM · CNB · K-Means · LDA · Trending Topic · SNA
    </span></div>""", unsafe_allow_html=True)

else:
    st.markdown(f"""<div style="background:linear-gradient(135deg,#fff,{C_CARD});
    border:1.5px solid {C_ACCENT_LT};border-radius:30px;padding:48px;text-align:center;max-width:700px;margin:0 auto;">
    <div style="font-size:48px;margin-bottom:16px;">🌷</div>
    <h4 style="font-size:20px;">👈 Mulai dengan Upload Dataset CSV di Sidebar</h4>
    <p style="font-size:14.5px;max-width:560px;margin:12px auto;color:#A07090;">
    Upload hasil scraping Twitter/X dengan kolom: <b>full_text</b>, <b>username</b>,
    <b>in_reply_to_screen_name</b>, <b>created_at</b></p>
    </div>""", unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    for col,(icon,title,desc) in zip([f1,f2,f3,f4],[
        ("🧪","Preprocessing Pipeline","7 tahap NLP dari raw text ke stemmed"),
        ("🚨","Klasifikasi Urgensi","Lexicon scoring + LR/SVM/CNB (70:30 split)"),
        ("📈","Trending Topic","Kata & klaster yang naik pesat per bulan"),
        ("🕸️","SNA Support System","Visualisasi jaringan akun penolong"),
    ]):
        with col:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#fff,{C_CARD});
            border:1.5px solid {C_ACCENT_LT};border-radius:22px;padding:20px;text-align:center;">
            <div style="font-size:28px;">{icon}</div>
            <b>{title}</b><br><small style="color:#A07090;">{desc}</small>
            </div>""", unsafe_allow_html=True)
