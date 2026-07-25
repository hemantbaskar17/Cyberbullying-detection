import streamlit as st
import joblib, json, re, numpy as np, time
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for r in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    nltk.download(r, quiet=True)

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
:root{
  --bg:#0f1117; --surf:#1a1d27; --surf2:#22263a;
  --bdr:#2e3348; --safe:#00d68f; --danger:#ff4d6d;
  --muted:#7b82a0; --text:#e8eaf0;
}
html,body,[class*="css"]{
  font-family:'DM Sans',sans-serif;
  background:var(--bg)!important;
  color:var(--text)!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2rem 4rem!important;max-width:700px!important;}
.hero{text-align:center;padding:2rem 0 1.5rem;}
.hero h1{
  font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
  background:linear-gradient(135deg,#6c63ff,#ff6584);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:.25rem;
}
.hero p{color:var(--muted);font-size:.88rem;}
.met{
  background:var(--surf);border:1px solid var(--bdr);
  border-radius:12px;padding:.9rem;text-align:center;
}
.met-val{
  font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
  background:linear-gradient(135deg,#6c63ff,#ff6584);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.met-lbl{color:var(--muted);font-size:.73rem;margin-top:.1rem;}
textarea{
  background:var(--surf2)!important;border:1.5px solid var(--bdr)!important;
  border-radius:12px!important;color:var(--text)!important;
  font-family:'DM Sans',sans-serif!important;font-size:.9rem!important;
}
textarea:focus{
  border-color:#6c63ff!important;
  box-shadow:0 0 0 3px rgba(108,99,255,.12)!important;
}
.stButton>button{
  width:100%;background:linear-gradient(135deg,#6c63ff,#8b5cf6)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-family:'Syne',sans-serif!important;font-size:.95rem!important;
  font-weight:700!important;padding:.7rem!important;transition:all .2s!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 6px 20px rgba(108,99,255,.35)!important;
}
.card-danger{
  background:rgba(255,77,109,.06);border:1.5px solid rgba(255,77,109,.3);
  border-radius:14px;padding:1.3rem 1.5rem;
  animation:pop .3s cubic-bezier(.34,1.56,.64,1) both;
}
.card-safe{
  background:rgba(0,214,143,.06);border:1.5px solid rgba(0,214,143,.3);
  border-radius:14px;padding:1.3rem 1.5rem;
  animation:pop .3s cubic-bezier(.34,1.56,.64,1) both;
}
.clabel{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;margin-bottom:.2rem;}
.csub{color:var(--muted);font-size:.82rem;margin-bottom:.8rem;}
.barlbl{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted);margin-bottom:.28rem;}
.barbg{background:var(--surf2);border-radius:999px;height:7px;overflow:hidden;}
.bar-d{height:100%;border-radius:999px;background:linear-gradient(90deg,#ff4d6d,#cc3355);}
.bar-s{height:100%;border-radius:999px;background:linear-gradient(90deg,#00d68f,#00b574);}
.chips{display:flex;gap:.45rem;margin-top:.75rem;}
.chip{
  background:var(--surf2);border:1px solid var(--bdr);
  border-radius:7px;padding:.25rem .6rem;font-size:.73rem;color:var(--muted);
}
.chip b{color:var(--text);}
.htitle{
  font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700;
  color:var(--muted);letter-spacing:1px;text-transform:uppercase;
  margin:1.6rem 0 .6rem;
}
.hitem{
  background:var(--surf);border:1px solid var(--bdr);border-radius:9px;
  padding:.6rem .9rem;margin-bottom:.38rem;
  display:flex;align-items:center;gap:.65rem;
}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.dot-d{background:var(--danger);}
.dot-s{background:var(--safe);}
.htxt{flex:1;font-size:.83rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hbadge{font-size:.72rem;color:var(--muted);flex-shrink:0;}
.footer{
  text-align:center;color:var(--muted);font-size:.73rem;
  margin-top:2rem;padding-top:1rem;border-top:1px solid var(--bdr);
}
@keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_all():
    model     = joblib.load('cyberbullying_model.pkl')
    tfidf     = joblib.load('tfidf_vectorizer.pkl')
    info      = json.load(open('model_info.json'))
    slang_df  = pd.read_csv('slang_dictionary.csv')
    slang     = dict(zip(slang_df['slang'].str.lower(),
                         slang_df['expansion']))
    neg_slang = set(slang_df[slang_df['type']=='negative']['slang'].str.lower())
    return model, tfidf, info, slang, neg_slang

try:
    model, tfidf, info, SLANG, NEG_SLANG = load_all()
except Exception as e:
    st.error("""Model files not found. Place these files in the same folder as app.py:
    - cyberbullying_model.pkl
    - tfidf_vectorizer.pkl
    - model_info.json
    - slang_dictionary.csv""")
    st.stop()


# NLP Setup
lem = WordNetLemmatizer()
sw  = set(stopwords.words('english')) - {
      'no','not','never','nobody','nothing','nowhere','nor','neither'}

NEGATIVE_INDICATORS = {
    'hate','kill','die','ugly','stupid','dumb','idiot','moron','pathetic',
    'worthless','useless','fat','loser','freak','disgusting','horrible',
    'terrible','awful','garbage','creep','weirdo','psycho','harass',
    'attack','hurt','destroy','abuse','racist','sexist','coward',
    'failure','pig','fuck','bitch','asshole','bastard','shit','jerk',
    'scum','pervert','slut','whore','retard','scumbag','dumbass',
    'trash','murderer','criminal','monster','evil','filth','crap',
}
NEGATIVE_INDICATORS.update(NEG_SLANG)

POSITIVE_INDICATORS = {
    'love','great','amazing','wonderful','awesome','good','nice','beautiful',
    'kind','sweet','happy','proud','excellent','fantastic','brilliant',
    'smart','cool','fun','thanks','thank','congrats','welcome',
    'appreciate','respect','support','help','hope','care','friend',
    'incredible','cheerful','blessed','grateful','inspiring',
}

THREAT_PATTERNS = [
    r'\bkys\b', r'\bkill\s*your\s*self\b', r'\bgo\s*die\b',
    r'\bnobody\s*(likes?|loves?|wants?|cares?)\s*(you|u)\b',
    r'\byou\s*deserve\s*to\s*(die|suffer|rot)\b',
    r'\bkill\s*(u|you)\b',
    r'\bi\s*(will|gonna|am\s*going\s*to)\s*(kill|hurt|destroy|murder)\s*(you|u)\b',
    r'\bfuck\s*(you|u|off)\b',
    r'\byou\s*(are|r)\s*(a\s*)?(bitch|asshole|bastard|whore|slut)\b',
    r'\bgo\s*to\s*hell\b',
    r'\bshut\s*(the\s*fuck\s*)?up\b',
    r'\byou\s*(stupid|dumb|ugly|fat|worthless|pathetic)\b',
    r'\byou\s*(little\s*)?(bitch|asshole|bastard|idiot|moron)\b',
    r'\b(nigga|nigger|faggot|fag|chink|spic|kike|paki|retard)\b',
]

def context_score(text):
    t     = text.lower()
    words = set(re.findall(r'\b\w+\b', t))
    for p in THREAT_PATTERNS:
        if re.search(p, t): return 1.0
    neg       = len(words & NEGATIVE_INDICATORS)
    pos       = len(words & POSITIVE_INDICATORS)
    neg_slang = len(words & NEG_SLANG)
    score     = (neg * 0.30 + neg_slang * 0.50 - pos * 0.20)
    return round(max(0.0, min(1.0, score / 3.0)), 4)

def preprocess(text):
    if not isinstance(text, str) or not text.strip(): return ''
    t = text.lower()
    t = re.sub(r'http\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = ' '.join([SLANG.get(w, w) for w in t.split()])
    t = re.sub(r"[^\w\s']", ' ', t)
    t = re.sub(r'\b\d+\b', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    tok = [lem.lemmatize(w) for w in word_tokenize(t)
           if w not in sw and len(w) > 1]
    return ' '.join(tok)

def predict(text):
    # Layer 1 — direct threat
    for p in THREAT_PATTERNS:
        if re.search(p, text.lower()):
            return 1, 99.0, 1.0
    # Layer 2 — ML
    ctx  = context_score(text)
    vec  = hstack([tfidf.transform([preprocess(text)]),
                   csr_matrix([[ctx]])])
    pred = model.predict(vec)[0]
    if hasattr(model, 'predict_proba'):
        conf = round(float(max(model.predict_proba(vec)[0]))*100, 1)
    elif hasattr(model, 'decision_function'):
        s    = model.decision_function(vec)[0]
        conf = round(float(1/(1+np.exp(-abs(s))))*100, 1)
    else:
        conf = 80.0
    # Layer 3 — override false positives
    if pred == 1 and ctx < 0.05:
        pred, conf = 0, round(conf*0.4, 1)
    return int(pred), conf, ctx


# UI

st.markdown("""
<div class="hero">
  <div style="font-size:2.8rem;margin-bottom:.35rem">🛡️</div>
  <h1>Cyberbullying Detector</h1>
  <p>Cyberbullying Detection on Social Media &nbsp;·&nbsp; NLP Project</p>
</div>
""", unsafe_allow_html=True)

# Metric cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="met"><div class="met-val">{info["accuracy"]*100:.1f}%</div><div class="met-lbl">Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="met"><div class="met-val">{info["f1"]*100:.1f}%</div><div class="met-lbl">F1 Score</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="met"><div class="met-val">{info["best_model"].split()[0]}</div><div class="met-lbl">Best Model</div></div>', unsafe_allow_html=True)

st.divider()

# Input
text_in = st.text_area("", placeholder="Enter or paste a social media message here...",
                        height=110, label_visibility="collapsed")
btn = st.button("🔍  Analyse Message")

if 'hist' not in st.session_state:
    st.session_state.hist = []

# Prediction
if btn:
    if not text_in.strip():
        st.warning("Please enter a message before analysing.")
    else:
        with st.spinner("Analysing..."):
            time.sleep(0.3)
            pred, conf, ctx = predict(text_in.strip())

        harmful = pred == 1
        if harmful:
            st.markdown(f"""
            <div class="card-danger">
              <div class="clabel">🚨 Cyberbullying Detected</div>
              <div class="csub">This message contains harmful or abusive content.</div>
              <div class="barlbl"><span>Confidence</span><span>{conf}%</span></div>
              <div class="barbg"><div class="bar-d" style="width:{conf}%"></div></div>
              <div class="chips">
                <div class="chip">Context Score <b>{ctx}</b></div>
                <div class="chip">Word Count <b>{len(text_in.split())}</b></div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-safe">
              <div class="clabel">✅ Not Cyberbullying</div>
              <div class="csub">This message appears to be safe and non-harmful.</div>
              <div class="barlbl"><span>Confidence</span><span>{conf}%</span></div>
              <div class="barbg"><div class="bar-s" style="width:{conf}%"></div></div>
              <div class="chips">
                <div class="chip">Context Score <b>{ctx}</b></div>
                <div class="chip">Word Count <b>{len(text_in.split())}</b></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.session_state.hist.insert(0, {
            't': text_in.strip(), 'h': harmful, 'c': f"{conf}%"
        })
        st.session_state.hist = st.session_state.hist[:6]

# History
if st.session_state.hist:
    st.markdown("<div class='htitle'>Recent Analyses</div>", unsafe_allow_html=True)
    for h in st.session_state.hist:
        dot   = "dot-d" if h['h'] else "dot-s"
        badge = "Harmful" if h['h'] else "Safe"
        txt   = h['t'][:65]+"…" if len(h['t'])>65 else h['t']
        st.markdown(f"""
        <div class="hitem">
          <div class="dot {dot}"></div>
          <div class="htxt">{txt}</div>
          <div class="hbadge">{badge} · {h['c']}</div>
        </div>""", unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.hist = []
        st.rerun()

st.markdown("""
<div class="footer">
  🛡️ Cyberbullying Detector &nbsp;·&nbsp; developed by Usual Suspects &nbsp;·&nbsp;
  Chennai Institute of Technology &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)