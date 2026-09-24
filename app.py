import streamlit as st
import pandas as pd
import numpy as np
import re, os, io, json, math, unicodedata
from datetime import datetime, date
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
try:
    from postgrest.exceptions import APIError
except ImportError:
    APIError = Exception

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Linguistic Corpus Engine", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.2em; font-weight: bold; color: #1f77b4; margin-bottom: 0.3em; }
    .subheader { font-size: 1.15em; font-weight: bold; color: #333; margin-top: 0.8em; margin-bottom: 0.4em; }
    .kwic-node { font-weight: bold; background-color: #fff3cd; padding: 0.1em 0.35em; border-radius: 0.25em; }
    .ts-chip { font-size: 0.85em; color: #555; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing Supabase credentials. Add SUPABASE_URL and SUPABASE_KEY to your secrets (local: .streamlit/secrets.toml, cloud: Streamlit Cloud app settings).")
        st.stop()
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Could not connect to Supabase: {e}")
        st.stop()

supabase = init_supabase()

# ============================================================
# DB HELPERS  (fix #1: paginate everything, REST caps at 1000 rows)
# ============================================================
PAGE = 1000

def fetch_all(table: str, select: str = "*", eq: dict | None = None, order: str | None = None, desc: bool = False, max_rows: int | None = None):
    q = supabase.table(table).select(select)
    for k, v in (eq or {}).items():
        if v is not None:
            q = q.eq(k, v)
    if order:
        q = q.order(order, desc=desc)
    out, off = [], 0
    while True:
        r = q.range(off, off + PAGE - 1).execute()
        data = r.data or []
        out.extend(data)
        off += PAGE
        if len(data) < PAGE or (max_rows and len(out) >= max_rows):
            break
    return out

@st.cache_data(ttl=300, show_spinner=False)
def cached_corpora():
    return fetch_all("corpora", order="created_at", desc=True)

@st.cache_data(ttl=300, show_spinner=False)
def cached_transcripts(corpus_id):
    return fetch_all("transcriptions", select="id,title,language,video_url,publish_date,channel,duration_seconds,speaker,genre,raw_text,clean_text,lemma_counts,word_counts,pos_counts,created_at", eq={"corpus_id": corpus_id}, order="created_at", desc=True)

@st.cache_data(ttl=300, show_spinner=False)
def cached_metadata_fields():
    return fetch_all("metadata_fields", order="id")

@st.cache_data(ttl=300, show_spinner=False)
def cached_lexicon():
    return fetch_all("lexicon")

@st.cache_data(ttl=300, show_spinner=False)
def cached_anglicisms():
    return fetch_all("anglicisms")

def clear_caches():
    st.cache_data.clear()

HINTS = {
    "42501": "Row-level security rejected the write. Run schema_fix.sql in the Supabase SQL Editor (it re-creates the access policies and grants).",
    "PGRST204": "A column is missing from a table (usually the table was created earlier with an older definition). Run schema_migrate.sql in the Supabase SQL Editor - it adds all missing columns without deleting data. If the error still shows a column that does exist, reload the API schema: Supabase Dashboard > Settings > API > 'Reload schema' (or wait 1-2 minutes for the cache to refresh).",
    "PGRST205": "A table is missing in the database. Run schema.sql in the Supabase SQL Editor first.",
    "23505": ("A duplicate-name conflict: if it mentions 'corpora', that corpus name already exists - choose a different name. "
               "If it mentions 'anglicisms_term_key' or 'lexicon', it is a harmless background dictionary-seeding duplicate and can be ignored."),
    "23502": "A required field was empty.",
    "42P01": "The table does not exist. Run schema.sql in the Supabase SQL Editor first.",
    "PGRST301": "The API key was rejected. Check SUPABASE_KEY: it must be the anon public key from Supabase > Project Settings > API.",
}

def show_db_error(e):
    code = getattr(e, "code", "") or ""
    msg = getattr(e, "message", None) or str(e)
    st.error(f"Database error [{code or 'unknown'}]: {msg}")
    for k, hint in HINTS.items():
        if code == k or (not code and k in msg):
            st.info(hint)
            break

def db_insert(table, rows, chunk=900):
    """Insert rows in chunks; rows may be a dict or list of dicts. Returns inserted data ([] on failure)."""
    if isinstance(rows, dict):
        rows = [rows]
    inserted = []
    for i in range(0, len(rows), chunk):
        try:
            r = supabase.table(table).insert(rows[i:i+chunk]).execute()
            inserted.extend(r.data or [])
        except APIError as e:
            show_db_error(e)
            return []
        except Exception as e:
            st.error(f"Database error: {e}")
            return []
    return inserted

def db_delete(table, column, value):
    try:
        supabase.table(table).delete().eq(column, value).execute()
    except APIError as e:
        show_db_error(e)
    except Exception as e:
        st.error(f"Database error: {e}")

# ============================================================
# NLP (fix #6: spaCy models come from requirements.txt; graceful fallback)
# ============================================================
SPACY_MODELS = {"pl": "pl_core_news_sm", "en": "en_core_web_sm", "de": "de_core_web_sm", "fr": "fr_core_web_sm", "es": "es_core_web_sm", "zh": "zh_core_web_sm"}

@st.cache_resource
def load_spacy_model(lang_code: str):
    model_name = SPACY_MODELS.get(lang_code, "en_core_web_sm")
    try:
        import spacy
        try:
            return spacy.load(model_name), model_name
        except OSError:
            st.info(f" spaCy model '{model_name}' is not installed in this environment. Falling back to the built-in regex tokenizer (no POS tags). Add '{model_name}' to requirements.txt for full tagging.")
            return None, model_name
    except ImportError:
        return None, model_name

WORD_RE = re.compile(r"\w+", re.UNICODE)

def tokenize_fallback(text: str):
    toks = []
    for i, m in enumerate(WORD_RE.finditer(text)):
        w = m.group(0)
        toks.append({"token_index": i, "word": w, "lemma": w.lower(), "pos": "UNK", "tag": "UNK"})
    return toks

def tokenize(text: str, lang: str):
    nlp, _ = load_spacy_model(lang)
    if nlp is None:
        return tokenize_fallback(text)
    # keep memory sane on very long transcripts
    doc = nlp(text if len(text) < 90000 else text[:90000])
    return [{"token_index": i, "word": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_}
            for i, t in enumerate(doc) if not t.is_space]

# ============================================================
# TEXT CLEANING + TIMESTAMP PARSING (fix #4)
# ============================================================
TS_INLINE_RE = re.compile(r"[\[(](\d{1,2}:)?(\d{1,2}):(\d{2})[\])]")
CUE_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,]\d{1,3}\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,]\d{1,3}")
VTT_CUE_RE = re.compile(r"(\d{1,2}):(\d{2})[.:](\d{2})[.,]\d*\s*-->\s*(\d{1,2}):(\d{2})[.:](\d{2})[.,]\d*")

def _hms(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)

def clean_and_normalize(text: str) -> str:
    text = re.sub(r"WEBVTT.*?\n", "", text)
    text = TS_INLINE_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip vtt/srt markup tags
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_timed_segments(raw: str):
    """Return list of (segment_text, ts_start_seconds_or_None). Supports SRT/VTT cues and inline [mm:ss] markers."""
    if "-->" in raw:
        segments, cur_start, cur_lines = [], None, []
        for line in raw.splitlines():
            m = CUE_RE.search(line) or VTT_CUE_RE.search(line)
            if m:
                if cur_lines and cur_start is not None:
                    segments.append((" ".join(cur_lines), cur_start))
                cur_start = _hms(m.group(1), m.group(2), m.group(3))
                cur_lines = []
            elif line.strip() and not re.fullmatch(r"\d+", line.strip()) and "WEBVTT" not in line:
                cur_lines.append(line.strip())
        if cur_lines and cur_start is not None:
            segments.append((" ".join(cur_lines), cur_start))
        if segments:
            return segments
    if TS_INLINE_RE.search(raw):
        segments, last, buf = [], 0.0, []
        pos = 0
        for m in TS_INLINE_RE.finditer(raw):
            pre = raw[pos:m.start()]
            if pre.strip():
                buf.append(pre)
            if buf:
                segments.append((" ".join(buf), last))
            h = m.group(1)
            last = _hms(h[:-1] if h else 0, m.group(2), m.group(3))
            buf, pos = [], m.end()
        if raw[pos:].strip() or buf:
            buf.append(raw[pos:])
            segments.append((" ".join(buf), last))
        if segments:
            return segments
    return [(raw, None)]

# ============================================================
# INGEST
# ============================================================
def build_counts(tokens):
    lemma_c = Counter(t["lemma"].lower() for t in tokens if t["lemma"].strip())
    word_c = Counter(t["word"].lower() for t in tokens if t["word"].strip())
    pos_c = Counter(t["pos"] for t in tokens)
    return lemma_c, word_c, pos_c

def save_transcript(corpus_id: int, title: str, lang: str, raw_text: str, metadata: dict | None = None, video_url: str = ""):
    try:
        segments = parse_timed_segments(raw_text)
        clean_txt = clean_and_normalize(" ".join(s for s, _ in segments))
        row = {
            "corpus_id": corpus_id, "title": title, "language": lang,
            "raw_text": raw_text, "clean_text": clean_txt, "video_url": video_url or None,
            "created_at": datetime.now().isoformat(),
        }
        if metadata:
            for k, v in metadata.items():
                if v not in (None, ""):
                    row[k] = v
        res = db_insert("transcriptions", row)
        if not res:
            st.error("Failed to insert transcript (check that schema.sql was run).")
            return 0
        t_id = res[0]["id"]

        token_rows = []
        idx_base = 0
        all_tokens = []
        with st.spinner("Tokenizing and tagging..."):
            for seg_text, ts in segments:
                seg_clean = clean_and_normalize(seg_text)
                toks = tokenize(seg_clean, lang)
                for t in toks:
                    t["token_index"] = idx_base + t["token_index"]
                    t["ts_start"] = float(ts) if ts is not None else 0.0
                    all_tokens.append(t)
                idx_base += len(toks)
        for t in all_tokens:
            token_rows.append({
                "corpus_id": corpus_id, "transcript_id": t_id, "token_index": t["token_index"],
                "word": t["word"].lower(), "lemma": t["lemma"].lower(),
                "pos": t["pos"], "ts_start": t["ts_start"],
            })
        for i in range(0, len(token_rows), 1000):
            supabase.table("tokens").insert(token_rows[i:i+1000]).execute()

        lemma_c, word_c, pos_c = build_counts(all_tokens)
        supabase.table("transcriptions").update({
            "lemma_counts": lemma_c, "word_counts": word_c, "pos_counts": pos_c,
        }).eq("id", t_id).execute()
        clear_caches()
        return len(token_rows)
    except Exception as e:
        st.error(f"Error saving transcript: {e}")
        return 0

def delete_transcript(t_id):
    db_delete("tokens", "transcript_id", t_id)
    db_delete("transcriptions", "id", t_id)
    clear_caches()

def delete_corpus(corpus_id):
    db_delete("tokens", "corpus_id", corpus_id)
    db_delete("transcriptions", "corpus_id", corpus_id)
    db_delete("corpora", "id", corpus_id)
    clear_caches()

def update_transcript(t_id, updates: dict):
    supabase.table("transcriptions").update(updates).eq("id", t_id).execute()
    clear_caches()

def corpus_word_freq(corpus_id, level="word"):
    """Aggregate lemma/word counts across the corpus from the jsonb layer (fast, no full token scan)."""
    col = "lemma_counts" if level == "lemma" else "word_counts"
    total = Counter()
    for t in cached_transcripts(corpus_id):
        counts = t.get(col) or {}
        total.update({k: int(v) for k, v in counts.items()})
    return total

def corpus_size(corpus_id):
    n = 0
    for t in cached_transcripts(corpus_id):
        n += sum(int(v) for v in (t.get("word_counts") or {}).values())
    return n

# ============================================================
# SEED DATA (metadata schema, lexicons, anglicisms) - idempotent
# ============================================================
DEFAULT_METADATA_FIELDS = [
    ("video_id", "Video ID", "text", True),
    ("channel", "Channel / Source", "text", True),
    ("publish_date", "Publish date", "date", True),
    ("duration_seconds", "Duration (seconds)", "number", True),
    ("video_url", "Video URL", "text", True),
    ("speaker", "Main speaker(s)", "text", False),
    ("genre", "Genre", "select", False),
    ("language", "Language", "text", False),
]
DEFAULT_GENRES = ["interview", "podcast", "vlog", "news", "lecture", "live stream", "other"]

DEFAULT_LEXICON = [
    # Polish sentiment
    ("dobry", "pl", "positive", 1), ("świetny", "pl", "positive", 1), ("super", "pl", "positive", 1),
    ("wspaniały", "pl", "positive", 1), ("cieszę", "pl", "positive", 1), ("lubię", "pl", "positive", 1),
    ("dobra", "pl", "positive", 1), ("dobrze", "pl", "positive", 1), ("zgoda", "pl", "positive", 1),
    ("zły", "pl", "negative", -1), ("zła", "pl", "negative", -1), ("katastrofa", "pl", "negative", -1),
    ("straszny", "pl", "negative", -1), ("nienawidzę", "pl", "negative", -1), ("problem", "pl", "negative", -1),
    ("problem", "en", "negative", -1), ("crisis", "en", "negative", -1),
    ("great", "en", "positive", 1), ("good", "en", "positive", 1), ("love", "en", "positive", 1),
    ("amazing", "en", "positive", 1), ("bad", "en", "negative", -1), ("terrible", "en", "negative", -1),
    ("awful", "en", "negative", -1), ("hate", "en", "negative", -1),
    # modality: epistemic
    ("chyba", "pl", "epistemic", 0), ("może", "pl", "epistemic", 0), ("pewnie", "pl", "epistemic", 0),
    ("prawdopodobnie", "pl", "epistemic", 0), ("wydaje mi się", "pl", "epistemic", 0),
    ("maybe", "en", "epistemic", 0), ("probably", "en", "epistemic", 0), ("perhaps", "en", "epistemic", 0), ("seems", "en", "epistemic", 0),
    # modality: deontic
    ("musisz", "pl", "deontic", 0), ("musimy", "pl", "deontic", 0), ("trzeba", "pl", "deontic", 0),
    ("powinieneś", "pl", "deontic", 0), ("powinno się", "pl", "deontic", 0), ("wolno", "pl", "deontic", 0),
    ("must", "en", "deontic", 0), ("should", "en", "deontic", 0), ("have to", "en", "deontic", 0), ("need to", "en", "deontic", 0),
    # stance markers
    ("zgadzam się", "pl", "stance_agree", 1), ("nie zgadzam się", "pl", "stance_disagree", -1),
    ("uważam, że", "pl", "stance_agree", 1), ("według mnie", "pl", "stance_agree", 1),
    ("absurd", "pl", "stance_disagree", -1), ("bzdura", "pl", "stance_disagree", -1),
    ("i agree", "en", "stance_agree", 1), ("i disagree", "en", "stance_disagree", -1),
    ("in my opinion", "en", "stance_agree", 1), ("nonsense", "en", "stance_disagree", -1),
]

DEFAULT_ANGLICISMS = [
    ("startup", "business"), ("lider", "business"), ("trend", "general"), ("trending", "social media"),
    ("like", "social media"), ("share", "social media"), ("follow", "social media"), ("story", "social media"),
    ("stream", "tech"), ("streaming", "tech"), ("gaming", "tech"), ("hardware", "tech"), ("software", "tech"),
    ("app", "tech"), ("bug", "tech"), ("update", "tech"), ("download", "tech"), ("upload", "tech"),
    ("content", "media"), ("influencer", "media"), ("content creator", "media"), ("vlog", "media"),
    ("challenge", "general"), ("team", "general"), ("meeting", "business"), ("deadline", "business"),
    ("brainstorming", "business"), ("networking", "business"), ("ok", "general"), ("oops", "general"),
    ("wow", "general"), ("sorry", "general"), ("cool", "general"), ("wow", "interjection"),
    ("fitness", "lifestyle"), ("weekend", "lifestyle"), ("fast food", "lifestyle"), ("shopping", "lifestyle"),
    ("show", "media"), ("hit", "media"), ("hit", "music"), ("remix", "music"), ("cover", "music"),
]

def ensure_seed_data():
    """Insert only missing defaults. Avoid ON CONFLICT because legacy tables may
    not have the matching unique constraints. Existing data is preserved."""
    changed = False

    def seed_row(table, row):
        nonlocal changed
        try:
            result = supabase.table(table).insert(row).execute()
            changed = changed or bool(result.data)
            return True
        except Exception as e:
            code = getattr(e, "code", None)
            # A concurrent app session or an existing unique constraint can win
            # the race after our pre-check. Treat that as an already-seeded row.
            if code == "23505":
                return True
            # Surface actual setup issues, but do not halt browsing/searching.
            st.warning(f"Default {table} data could not be seeded ({code or 'database error'}): {getattr(e, 'message', str(e))}")
            return False

    try:
        existing_fields = {f["field_name"] for f in fetch_all("metadata_fields", select="field_name")}
        for name, label, ftype, mandatory in DEFAULT_METADATA_FIELDS:
            if name not in existing_fields:
                seed_row("metadata_fields", {"field_name": name, "label": label, "field_type": ftype,
                          "mandatory": mandatory, "options": DEFAULT_GENRES if ftype == "select" else None})
                existing_fields.add(name)

        existing_lex = {(x["term"].lower(), x.get("lang", "pl"), x["category"])
                        for x in fetch_all("lexicon", select="term,lang,category")}
        for term, lang, category, value in DEFAULT_LEXICON:
            key = (term.lower(), lang, category)
            if key not in existing_lex:
                if seed_row("lexicon", {"term": term, "lang": lang, "category": category, "value": value}):
                    existing_lex.add(key)

        existing_ang = {x["term"].lower() for x in fetch_all("anglicisms", select="term")}
        for term, category in DEFAULT_ANGLICISMS:
            key = term.lower()
            if key not in existing_ang:
                if seed_row("anglicisms", {"term": term, "category": category}):
                    existing_ang.add(key)

        if changed:
            clear_caches()
    except Exception as e:
        code = getattr(e, "code", None)
        st.warning(f"Default data seeding could not complete ({code or 'database error'}): {getattr(e, 'message', str(e))}. Run schema_migrate.sql in Supabase.")

POLISH_STOP = set("i w na z z do o a że nie się to jest jak co ale or oraz by dla pod nad za od przy przez który która które czym gdy gdyż więc czyli też już jeszcze bardzo tylko nawet tam tu tuż no well oraz albo lub niż bez niż".split())
POLISH_DIACRITICS = set("ąćęłńóśźż")

# ============================================================
# ANALYSIS HELPERS
# ============================================================
def log_likelihood(f1, n1, f2, n2):
    """LL keyness, corpus 1 vs corpus 2 (Rayson & Garside)."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (f1 + f2) / (n1 + n2)
    e1, e2 = n1 * p, n2 * p
    ll = 0.0
    if f1 > 0: ll += f1 * math.log(f1 / e1)
    if f2 > 0: ll += f2 * math.log(f2 / e2)
    return 2 * ll

def slice_transcripts(transcripts, granularity):
    """Assign each transcript to a time slice based on publish_date."""
    rows = []
    for t in transcripts:
        d = t.get("publish_date")
        if not d:
            continue
        d = str(d)[:10]
        if granularity == "Year":
            key = d[:4]
        elif granularity == "Quarter":
            y, m = d[:4], int(d[5:7])
            key = f"{y}-Q{(m - 1) // 3 + 1}"
        else:
            key = d[:7]
        rows.append({"id": t["id"], "date": d, "slice": key, "lemma_counts": t.get("lemma_counts") or {}, "word_counts": t.get("word_counts") or {}})
    return pd.DataFrame(rows)

def transcripts_with_word(corpus_id, word, level="word"):
    col = "word_counts" if level == "word" else "lemma_counts"
    return [t for t in cached_transcripts(corpus_id) if word in (t.get(col) or {})]

def youtube_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else None

def fmt_ts(seconds):
    if seconds is None or seconds <= 0:
        return ""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"
# ============================================================
# SIDEBAR: corpus management
# ============================================================
with st.sidebar:
    st.markdown("## Corpus Management")
    st.session_state.setdefault("corpus_id", None)

    corpora = cached_corpora()
    if not corpora:
        st.info("No corpora yet. Create one below.")
        st.session_state.corpus_id = None
    else:
        labels = {c["id"]: c["name"] for c in corpora}
        current = st.session_state.corpus_id if st.session_state.corpus_id in labels else list(labels)[0]
        sel = st.selectbox("Active corpus", options=list(labels), format_func=lambda i: labels[i], index=list(labels).index(current))
        st.session_state.corpus_id = sel
        if st.button("Delete this corpus", type="secondary", use_container_width=True):
            st.session_state.confirm_delete = True
        if st.session_state.get("confirm_delete"):
            if st.checkbox("I understand this deletes all texts and tokens"):
                if st.button("Confirm delete", type="primary"):
                    delete_corpus(sel)
                    st.session_state.corpus_id = None
                    st.session_state.confirm_delete = False
                    st.rerun()

    st.divider()
    st.markdown("### Create new corpus")
    new_name = st.text_input("Name", key="new_corpus_name", placeholder="e.g. Polish YouTube Podcasts")
    new_desc = st.text_area("Description", key="new_corpus_desc", height=70)
    new_lang = st.selectbox("Main language", list(SPACY_MODELS), format_func=lambda l: {"pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}[l])
    if st.button("Create corpus", type="primary", use_container_width=True):
        if not new_name.strip():
            st.error("Enter a corpus name.")
        elif any(c["name"].lower() == new_name.strip().lower() for c in corpora):
            st.error(f"A corpus named '{new_name.strip()}' already exists. Pick another name.")
        else:
            created = db_insert("corpora", {"name": new_name.strip(), "description": new_desc, "language": new_lang, "created_at": datetime.now().isoformat(), "token_count": 0})
            if created:
                clear_caches()
                st.success(f"Corpus '{new_name.strip()}' created.")
                st.rerun()

corpus_id = st.session_state.get("corpus_id")
if not corpus_id:
    st.markdown("<h1 class='main-header'>Linguistic Corpus Engine</h1>", unsafe_allow_html=True)
    st.warning("Create or select a corpus in the sidebar to start.")
    st.stop()

ensure_seed_data()
transcripts = cached_transcripts(corpus_id)
CORPUS_LANG = next((c.get("language", "pl") for c in cached_corpora() if c["id"] == corpus_id), "pl")

# header metrics
total_tokens = corpus_size(corpus_id)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Texts", len(transcripts))
c2.metric("Tokens", f"{total_tokens:,}")
types = len(corpus_word_freq(corpus_id, "lemma"))
c3.metric("Types (lemmas)", f"{types:,}")
c4.metric("TTR", f"{types / total_tokens:.3f}" if total_tokens else "n/a")
st.divider()

# ============================================================
# TAB 1: UPLOAD & MANAGE (fix #3: edit + delete + metadata schema)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Upload & Manage", "KWIC + Timestamps", "Dictionaries", "Frequency & Trends",
    "Neologisms & Keyness", "Semantic Drift & Keywords over Time", "Loanwords & Anglicisms",
    "Stance / Sentiment / Modality", "Statistics & Metadata",
])

with tab1:
    st.markdown("<h3 class='subheader'>Add texts</h3>", unsafe_allow_html=True)
    meta_fields = cached_metadata_fields()
    upload_method = st.radio("Method:", ["Paste text", "Upload file (txt/srt/vtt)"], horizontal=True)

    def metadata_form(prefix):
        md = {}
        cols = st.columns(2)
        fidx = 0
        for f in meta_fields:
            if f["field_name"] in ("video_url", "language"):
                continue  # handled separately
            widget_key = f"{prefix}_{f['field_name']}"
            with cols[fidx % 2]:
                if f["field_type"] == "select":
                    opts = f.get("options") or []
                    v = st.selectbox(f["label"] + (" *" if f["mandatory"] else ""), opts + ["(none)"], key=widget_key)
                    if v not in (None, "(none)"):
                        md[f["field_name"]] = v
                elif f["field_type"] == "date":
                    v = st.date_input(f["label"] + (" *" if f["mandatory"] else ""), value=None, key=widget_key)
                    if v:
                        md[f["field_name"]] = v.isoformat()
                elif f["field_type"] == "number":
                    v = st.number_input(f["label"] + (" *" if f["mandatory"] else ""), min_value=0, step=1, key=widget_key)
                    if v is not None:
                        md[f["field_name"]] = int(v)
                else:
                    v = st.text_input(f["label"] + (" *" if f["mandatory"] else ""), key=widget_key)
                    if v.strip():
                        md[f["field_name"]] = v.strip()
            fidx += 1
        return md

    if upload_method == "Paste text":
        title = st.text_input("Title *", key="paste_title")
        video_url = st.text_input("Video URL (YouTube link enables the timestamp viewer)", key="paste_url")
        md = metadata_form("paste")
        missing = [f["label"] for f in meta_fields if f["mandatory"] and f["field_name"] not in md and f["field_name"] != "language"]
        text_input = st.text_area("Text content *", height=220, key="paste_text", placeholder="Paste transcript. Inline [mm:ss] markers or SRT/VTT cues are detected automatically for timestamps.")
        if st.button("Add to corpus", type="primary", use_container_width=True):
            if not title.strip() or not text_input.strip():
                st.error("Title and text are required.")
            elif missing:
                st.error("Missing mandatory metadata: " + ", ".join(missing))
            else:
                n = save_transcript(corpus_id, title.strip(), CORPUS_LANG, text_input, metadata=md, video_url=video_url)
                if n:
                    st.success(f"Saved. {n:,} tokens indexed.")
                    st.rerun()
    else:
        up = st.file_uploader("Upload .txt / .srt / .vtt (batch: select multiple)", type=["txt", "srt", "vtt"], accept_multiple_files=True)
        if up:
            md = metadata_form("file")
            missing = [f["label"] for f in meta_fields if f["mandatory"] and f["field_name"] not in md and f["field_name"] != "language"]
            if missing:
                st.warning("Mandatory fields still empty: " + ", ".join(missing))
            video_url = st.text_input("Video URL for these files (optional)", key="file_url")
            if st.button("Add files to corpus", type="primary", use_container_width=True):
                added = 0
                for f in up:
                    try:
                        content = f.read().decode("utf-8-sig", errors="replace")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                        continue
                    tname = re.sub(r"\.(txt|srt|vtt)$", "", f.name, flags=re.I)
                    n = save_transcript(corpus_id, tname, CORPUS_LANG, content, metadata=md, video_url=video_url)
                    added += 1 if n else 0
                if added:
                    st.success(f"Added {added} file(s).")
                    st.rerun()

    st.divider()
    st.markdown("<h3 class='subheader'>Texts in this corpus (edit / delete)</h3>", unsafe_allow_html=True)
    if transcripts:
        for t in transcripts:
            with st.expander(f"{t['title']}  ({(t.get('publish_date') or 'no date')} | {(t.get('channel') or 'no channel')})"):
                e_title = st.text_input("Title", value=t["title"], key=f"et_{t['id']}")
                e_url = st.text_input("Video URL", value=t.get("video_url") or "", key=f"eu_{t['id']}")
                e_text = st.text_area("Clean text", value=t.get("clean_text") or "", height=140, key=f"ex_{t['id']}")
                e_md = {}
                for f in meta_fields:
                    cur = t.get(f["field_name"])
                    if f["field_type"] == "select":
                        opts = (f.get("options") or []) + ["(none)"]
                        v = st.selectbox(f["label"], opts, index=opts.index(cur) if cur in opts else len(opts) - 1, key=f"em_{t['id']}_{f['field_name']}")
                        e_md[f["field_name"]] = None if v == "(none)" else v
                    elif f["field_type"] == "date":
                        v = st.date_input(f["label"], value=pd.to_datetime(cur).date() if cur else None, key=f"em_{t['id']}_{f['field_name']}")
                        e_md[f["field_name"]] = v.isoformat() if v else None
                    else:
                        e_md[f["field_name"]] = st.text_input(f["label"], value=str(cur) if cur else "", key=f"em_{t['id']}_{f['field_name']}")
                cA, cB, cC = st.columns(3)
                if cA.button("Save changes", key=f"sv_{t['id']}"):
                    updates = {"title": e_title, "video_url": e_url or None}
                    updates.update({k: v for k, v in e_md.items()})
                    retext = e_text != (t.get("clean_text") or "")
                    if retext:
                        updates["clean_text"] = e_text
                        updates["raw_text"] = e_text
                    update_transcript(t["id"], updates)
                    if retext:
                        n = save_transcript(corpus_id, e_title, CORPUS_LANG, e_text, metadata=updates, video_url=e_url)
                        delete_transcript(t["id"])  # old tokens
                        st.success("Text and tokens re-indexed.")
                    else:
                        st.success("Saved.")
                    st.rerun()
                if cB.button("Re-tokenize", key=f"rt_{t['id']}"):
                    delete_transcript(t["id"])
                    save_transcript(corpus_id, t["title"], CORPUS_LANG, t.get("raw_text") or t.get("clean_text") or "", metadata={k: t.get(k) for k in [f["field_name"] for f in meta_fields]}, video_url=t.get("video_url") or "")
                    st.rerun()
                if cC.button("Delete", key=f"dl_{t['id']}", type="primary"):
                    delete_transcript(t["id"])
                    st.rerun()
    else:
        st.info("No texts yet.")

# ============================================================
# TAB 2: KWIC + TIMESTAMP VIEWER (fix #2: single query, fix #4: ts)
# ============================================================
with tab2:
    st.markdown("<h3 class='subheader'>KWIC concordance</h3>", unsafe_allow_html=True)
    kw_all = corpus_word_freq(corpus_id, "word")
    lm_all = corpus_word_freq(corpus_id, "lemma")
    if not kw_all:
        st.info("Upload texts first.")
    else:
        cL, cR = st.columns([3, 1])
        with cL:
            mode = st.radio("Match on:", ["word", "lemma"], horizontal=True, key="kwic_mode")
            query = st.text_input("Query (exact word, or wildcard with *):", placeholder="e.g. polityk* or 'jest'")
        with cR:
            window = st.slider("Context words", 2, 15, 5, key="kwic_win")
            max_hits = st.slider("Max hits", 10, 500, 100, key="kwic_max")
        pattern = "^" + re.escape(query.lower()).replace(r"\*", ".*") + "$" if query and "*" in query else (query.lower() if query else None)

        if st.button("Run KWIC", type="primary") and pattern:
            level_col = "word" if mode == "word" else "lemma"
            hits = []
            for t in transcripts:
                counts = t.get(f"{level_col}_counts") or {}
                if mode == "word":
                    total = sum(v for k, v in counts.items() if re.fullmatch(pattern, k))
                else:
                    total = sum(v for k, v in counts.items() if re.fullmatch(pattern, k))
                if total:
                    hits.append((t["id"], t["title"], total))
            hits.sort(key=lambda x: -x[2])
            st.info(f"{len(hits)} matching texts")
            hits = hits[:max_hits]
            if hits:
                rows = []
                for t_id, title, _ in hits:
                    toks = fetch_all("tokens", select="token_index,word,lemma,pos,ts_start", eq={"transcript_id": t_id}, order="token_index")
                    df = pd.DataFrame(toks)
                    if df.empty:
                        continue
                    words = df["word"].tolist()
                    for i, w in enumerate(words):
                        target = w if mode == "word" else df.iloc[i]["lemma"]
                        if re.fullmatch(pattern, target):
                            lo, hi = max(0, i - window), min(len(words), i + window + 1)
                            ts = df.iloc[i]["ts_start"]
                            rows.append({
                                "Text": title,
                                "Left": " ".join(words[lo:i]),
                                "Node": w.upper(),
                                "Right": " ".join(words[i + 1:hi]),
                                "POS": df.iloc[i]["pos"],
                                "Timestamp": fmt_ts(ts),
                                "token_index": i,
                                "transcript_id": t_id,
                            })
                            if len(rows) >= max_hits:
                                break
                    if len(rows) >= max_hits:
                        break
                kwic_df = pd.DataFrame(rows)
                st.dataframe(kwic_df[["Text", "Left", "Node", "Right", "POS", "Timestamp"]], use_container_width=True, hide_index=True)
                st.download_button("Download KWIC as CSV", kwic_df.drop(columns=["token_index", "transcript_id"]).to_csv(index=False), file_name="kwic.csv", mime="text/csv")

                # timestamp viewer
                st.divider()
                st.markdown("<h3 class='subheader'>Video timestamp viewer</h3>", unsafe_allow_html=True)
                url_for_ts = None
                for t in transcripts:
                    if t["id"] == kwic_df.iloc[0]["transcript_id"]:
                        url_for_ts = t.get("video_url")
                if url_for_ts:
                    yt = youtube_id(url_for_ts)
                else:
                    yt = None
                if not yt:
                    st.caption("Add a YouTube URL to a transcript to embed the player here.")
                else:
                    sel_row = st.selectbox("Jump to hit:", range(len(kwic_df)), format_func=lambda i: f"[{kwic_df.iloc[i]['Timestamp']}] {kwic_df.iloc[i]['Node']} in {kwic_df.iloc[i]['Text']}")
                    r = kwic_df.iloc[sel_row]
                    secs = None
                    ts_str = r["Timestamp"]
                    if ts_str:
                        parts = ts_str.split(":")
                        secs = int(parts[-1]) + 60 * int(parts[-2]) + (3600 * int(parts[-3]) if len(parts) == 3 else 0)
                    st.video(f"https://www.youtube.com/embed/{yt}?start={int(secs or 0)}&autoplay=0")
                    st.caption(f"Context: ... {r['Left']} **{r['Node']}** {r['Right']} ...")

# ============================================================
# TAB 3: DICTIONARIES (editable, stored in Supabase)
# ============================================================
with tab3:
    st.markdown("<h3 class='subheader'>Dictionaries (lexicon of sentiment / modality / stance)</h3>", unsafe_allow_html=True)
    st.caption("These entries are stored in the database and shared with everyone who opens the app. Add your own terms.")
    lex = cached_lexicon()
    lex_df = pd.DataFrame(lex)[["term", "lang", "category", "value"]] if lex else pd.DataFrame(columns=["term", "lang", "category", "value"])
    st.dataframe(lex_df, use_container_width=True, hide_index=True)
    with st.expander("Add dictionary entry"):
        n_term = st.text_input("Term / phrase", key="lex_term")
        n_lang = st.selectbox("Language", ["pl", "en"], key="lex_lang")
        n_cat = st.selectbox("Category", ["positive", "negative", "epistemic", "deontic", "stance_agree", "stance_disagree", "custom"], key="lex_cat")
        n_val = st.number_input("Value (for sentiment: -1..1)", -1.0, 1.0, 1.0, 0.5, key="lex_val")
        if st.button("Add entry", key="lex_add"):
            if n_term.strip():
                db_insert("lexicon", {"term": n_term.strip().lower(), "lang": n_lang, "category": n_cat, "value": float(n_val)})
                clear_caches()
                st.rerun()
    del_opts = [f"{r['term']} [{r['lang']}/{r['category']}]" for _, r in lex_df.iterrows()] if not lex_df.empty else []
    if del_opts and st.multiselect("Delete entries", del_opts, key="lex_del"):
        if st.button("Confirm delete", key="lex_del_go"):
            for sel_ in st.session_state.lex_del:
                term, rest = sel_.rsplit(" [", 1)
                lang, cat = rest.rstrip("]").split("/")
                row = next(l for l in lex if l["term"] == term and l["lang"] == lang and l["category"] == cat)
                db_delete("lexicon", "id", row["id"])
            clear_caches()
            st.rerun()

# ============================================================
# TAB 4: FREQUENCY & TRENDS (time slicing)
# ============================================================
with tab4:
    st.markdown("<h3 class='subheader'>Frequency & time-sliced trends</h3>", unsafe_allow_html=True)
    if not transcripts:
        st.info("Upload texts first.")
    else:
        f_mode = st.radio("Level:", ["word", "lemma"], horizontal=True, key="freq_mode")
        gran = st.selectbox("Time slice granularity", ["Year", "Quarter", "Month"], key="freq_gran")
        freq_df_all = corpus_word_freq(corpus_id, f_mode)
        top_terms = [w for w, _ in freq_df_all.most_common(40)]
        chosen = st.multiselect("Terms to track over time (empty = top 10 overall)", top_terms, default=[], key="freq_terms")
        if not chosen:
            chosen = [w for w, _ in freq_df_all.most_common(10)]
        sliced = slice_transcripts(transcripts, gran)
        if sliced.empty:
            st.warning("No publish_date values set. Fill the mandatory 'Publish date' metadata to enable trends.")
        else:
            series = {s: Counter() for s in sorted(sliced["slice"].unique())}
            sizes = {s: 0 for s in series}
            col_name = "lemma_counts" if f_mode == "lemma" else "word_counts"
            for _, r in sliced.iterrows():
                for k, v in (r[col_name] or {}).items():
                    series[r["slice"]][k] += int(v)
                    sizes[r["slice"]] += int(v)
            trend_rows = []
            for s in sorted(series):
                for term in chosen:
                    c = series[s].get(term, 0)
                    trend_rows.append({"slice": s, "term": term, "count": c,
                                       "per_million": round(c / sizes[s] * 1e6, 2) if sizes[s] else 0})
            tdf = pd.DataFrame(trend_rows)
            fig = px.line(tdf, x="slice", y="per_million", color="term", markers=True,
                          title=f"Frequency per million words ({gran.lower()}ly)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tdf.pivot(index="slice", columns="term", values="count").fillna(0), use_container_width=True)

        st.divider()
        n_show = st.slider("Show top N", 10, 100, 25, key="freq_top")
        freq_df = pd.DataFrame(freq_df_all.most_common(n_show), columns=[f_mode.capitalize(), "Frequency"])
        freq_df["Rel. freq (per 1M)"] = (freq_df["Frequency"] / total_tokens * 1e6).round(1) if total_tokens else 0
        st.plotly_chart(px.bar(freq_df.head(20), x=f_mode.capitalize(), y="Frequency", color="Frequency", color_continuous_scale="Blues", title=f"Most frequent {f_mode}s"), use_container_width=True)
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        st.download_button("Download full frequency list (CSV)", pd.DataFrame(freq_df_all.most_common(), columns=[f_mode, "freq"]).to_csv(index=False), file_name="frequencies.csv", mime="text/csv")
# ============================================================
# TAB 5: NEOLOGISMS & KEYNESS (reference corpus comparison)
# ============================================================
with tab5:
    st.markdown("<h3 class='subheader'>Neologism / emergence spotter & keyness</h3>", unsafe_allow_html=True)
    corpora_all = cached_corpora()
    other_opts = {c["id"]: c["name"] for c in corpora_all if c["id"] != corpus_id}
    if not other_opts:
        st.info("Create a second corpus (e.g. your small test corpus or a general reference corpus) to enable neologism spotting and keyness.")
    else:
        ref_id = st.selectbox("Reference corpus (baseline)", list(other_opts), format_func=lambda i: other_opts[i], key="key_ref")
        gran_neo = st.selectbox("Time slice for emergence", ["Year", "Quarter", "Month"], key="neo_gran")
        target_freq = corpus_word_freq(corpus_id, "word")
        ref_freq = corpus_word_freq(ref_id, "word")
        n_target = corpus_size(corpus_id)
        n_ref = corpus_size(ref_id)

        sliced_neo = slice_transcripts(transcripts, gran_neo)
        early, late = Counter(), Counter()
        n_early, n_late = 0, 0
        if not sliced_neo.empty:
            sl = sorted(sliced_neo["slice"].unique())
            col_name = "word_counts"
            half = max(1, len(sl) // 2)
            early_slices, late_slices = set(sl[:half]), set(sl[half:])
            for _, r in sliced_neo.iterrows():
                tgt = early if r["slice"] in early_slices else late
                for k, v in (r[col_name] or {}).items():
                    tgt[k] += int(v)
            n_early = sum(early.values())
            n_late = sum(late.values())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Emergent words (late vs early period)")
            if n_early and n_late:
                rows = []
                for w, f_late in late.most_common(300):
                    f_early = early.get(w, 0)
                    if f_late < 3:
                        continue
                    rate_e = f_early / n_early
                    rate_l = f_late / n_late
                    if rate_e == 0:
                        score = float("inf")
                        rows.append({"Word": w, "early": 0, "late": f_late, "emergence": "new"})
                    elif rate_l >= 2 * rate_e:
                        rows.append({"Word": w, "early": f_early, "late": f_late, "emergence": f"{rate_l / rate_e:.1f}x"})
                neo_df = pd.DataFrame(rows).head(25)
                st.dataframe(neo_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Needs publish_date metadata on at least two time slices.")
        with c2:
            st.markdown("#### Keyness (log-likelihood, this corpus vs reference)")
            min_freq = st.number_input("Min frequency in target corpus", 1, 50, 5, key="key_min")
            rows = []
            for w, f1 in target_freq.items():
                if f1 < min_freq:
                    continue
                f2 = ref_freq.get(w, 0)
                ll = log_likelihood(f1, n_target, f2, n_ref)
                if ll >= 6.63:  # p < 0.01
                    rows.append({"Word": w, "Target": f1, "Reference": f2, "LL": round(ll, 2)})
            key_df = pd.DataFrame(rows).sort_values("LL", ascending=False).head(40) if rows else pd.DataFrame()
            if not key_df.empty:
                st.dataframe(key_df, use_container_width=True, hide_index=True)
                st.download_button("Download keyness (CSV)", key_df.to_csv(index=False), file_name="keyness.csv", mime="text/csv")
            else:
                st.caption("No terms reached LL >= 6.63 (p < 0.01). Lower the min frequency.")

# ============================================================
# TAB 6: SEMANTIC DRIFT & KEYWORDS IN TIME
# ============================================================
with tab6:
    st.markdown("<h3 class='subheader'>Semantic drift & keyword-in-time tracker</h3>", unsafe_allow_html=True)
    gran_d = st.selectbox("Granularity", ["Year", "Quarter", "Month"], key="drift_gran")
    sliced_d = slice_transcripts(transcripts, gran_d)
    if sliced_d.empty:
        st.warning("Set publish_date metadata to use time tracking.")
    else:
        sl = sorted(sliced_d["slice"].unique())
        col_name = "word_counts"
        per_slice = {}
        sizes_d = {}
        for _, r in sliced_d.iterrows():
            per_slice.setdefault(r["slice"], Counter())
            for k, v in (r[col_name] or {}).items():
                per_slice[r["slice"]][k] += int(v)
            sizes_d[r["slice"]] = sizes_d.get(r["slice"], 0) + sum(int(v) for v in (r[col_name] or {}).values())

        st.markdown("#### Keywords in time (distinctive terms per slice)")
        global_freq = corpus_word_freq(corpus_id, "word")
        kw_rows = []
        for s in sl:
            for w, c in per_slice[s].most_common(15):
                share = c / max(1, sizes_d[s])
                glob = global_freq.get(w, 0) / max(1, total_tokens)
                if share >= 2 * glob and c >= 3:
                    kw_rows.append({"Slice": s, "Keyword": w, "Count": c, "per M": round(c / sizes_d[s] * 1e6, 1)})
        kw_df = pd.DataFrame(kw_rows)
        st.dataframe(kw_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Semantic drift: collocates of a word over time")
        drift_word = st.text_input("Word to track collocates for:", placeholder="e.g. europa", key="drift_word")
        span = st.slider("Collocate span (words each side)", 2, 10, 5, key="drift_span")
        if drift_word and st.button("Compute drift", key="drift_go"):
            t2t = {t["id"]: t for t in transcripts}
            drift_data = {}
            for s in sl:
                ids = sliced_d[sliced_d["slice"] == s]["id"].tolist()
                coll = Counter()
                for t_id in ids:
                    toks = fetch_all("tokens", select="word", eq={"transcript_id": t_id}, order="token_index")
                    words = [r["word"] for r in toks]
                    for i, w in enumerate(words):
                        if w == drift_word.lower():
                            for j in range(max(0, i - span), min(len(words), i + span + 1)):
                                if j != i and words[j] not in POLISH_STOP and len(words[j]) > 2:
                                    coll[words[j]] += 1
                drift_data[s] = coll
            drift_rows = []
            for s in sl:
                for w, c in drift_data[s].most_common(8):
                    drift_rows.append({"Slice": s, "Collocate": w, "Co-occurrences": c})
            drift_df = pd.DataFrame(drift_rows)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
            st.caption("Compare collocates across slices: changing collocates = semantic drift (e.g. 'europa' drifting from 'unia' to 'kryzys').")
            # heatmap
            top_coll = Counter()
            for c in drift_data.values():
                top_coll.update(c)
            top10 = [w for w, _ in top_coll.most_common(10)]
            hm = pd.DataFrame({s: {w: drift_data[s].get(w, 0) for w in top10} for s in sl})
            fig = px.imshow(hm, text_auto=True, aspect="auto", title=f"Collocates of '{drift_word}' across time slices")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 7: LOANWORDS & ANGLICISMS
# ============================================================
with tab7:
    st.markdown("<h3 class='subheader'>Loanword & anglicism tracker (Polish)</h3>", unsafe_allow_html=True)
    ang = cached_anglicisms()
    ang_df = pd.DataFrame(ang)[["term", "category"]] if ang else pd.DataFrame(columns=["term", "category"])
    st.caption("A stored list of English-origin items used in Polish. Extend it below; it persists for everyone.")
    with st.expander("Add anglicism"):
        a_term = st.text_input("Term", key="ang_term")
        a_cat = st.text_input("Domain (e.g. tech, social media)", key="ang_cat")
        if st.button("Add", key="ang_add") and a_term.strip():
            db_insert("anglicisms", {"term": a_term.strip().lower(), "category": a_cat.strip() or "general"})
            clear_caches()
            st.rerun()
    if not ang_df.empty:
        counts = []
        for _, r in ang_df.iterrows():
            c = corpus_word_freq(corpus_id, "word").get(r["term"], 0)
            counts.append({"Term": r["term"], "Domain": r["category"], "Count in corpus": c})
        ang_counts = pd.DataFrame(counts).sort_values("Count in corpus", ascending=False)
        shown = ang_counts[ang_counts["Count in corpus"] > 0]
        st.markdown(f"#### {len(shown)} anglicisms found in this corpus")
        st.plotly_chart(px.bar(shown.head(20), x="Term", y="Count in corpus", color="Domain", title="Anglicisms by frequency"), use_container_width=True)
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download anglicisms (CSV)", ang_counts.to_csv(index=False), file_name="anglicisms.csv", mime="text/csv")
        # per-slice trend of top anglicisms
        gran_a = st.selectbox("Time granularity for anglicism trends", ["Year", "Quarter", "Month"], key="ang_gran")
        sliced_a = slice_transcripts(transcripts, gran_a)
        if not sliced_a.empty and shown.head(5)["Term"].tolist():
            series_a = {}
            col_name = "word_counts"
            for _, r in sliced_a.iterrows():
                series_a.setdefault(r["slice"], Counter())
                for k, v in (r[col_name] or {}).items():
                    series_a[r["slice"]][k] += int(v)
            a_rows = []
            for s in sorted(series_a):
                for term in shown.head(5)["Term"]:
                    a_rows.append({"slice": s, "term": term, "per_million": round(series_a[s].get(term, 0) / max(1, sum(series_a[s].values())) * 1e6, 2)})
            st.plotly_chart(px.line(pd.DataFrame(a_rows), x="slice", y="per_million", color="term", markers=True, title="Top anglicisms over time"), use_container_width=True)

# ============================================================
# TAB 8: STANCE / SENTIMENT / MODALITY
# ============================================================
with tab8:
    st.markdown("<h3 class='subheader'>Stance, sentiment & modality filter</h3>", unsafe_allow_html=True)
    lex = cached_lexicon()
    if not lex:
        st.info("Add terms in the Dictionaries tab first.")
    else:
        sent = {l["term"]: l["value"] for l in lex if l["lang"] == "pl" and l["category"] in ("positive", "negative")}
        epi = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "epistemic"}
        deo = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "deontic"}
        agr = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "stance_agree"}
        dis = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "stance_disagree"}

        st.markdown("Filter transcripts by stance / sentiment / modality profile:")
        c1, c2, c3 = st.columns(3)
        want_sent = c1.selectbox("Sentiment tilt", ["any", "positive-lean", "negative-lean"], key="st_sent")
        want_epi = c2.slider("Min epistemic markers (hedging)", 0, 20, 0, key="st_epi")
        want_deo = c3.slider("Min deontic markers (obligation)", 0, 20, 0, key="st_deo")
        want_stance = c1.selectbox("Stance", ["any", "agree-lean", "disagree-lean"], key="st_stance")

        results = []
        for t in transcripts:
            words = {k.lower() for k in (t.get("word_counts") or {})}
            text_l = (t.get("clean_text") or "").lower()
            pos_n = sum(t.get("word_counts", {}).get(w, 0) for w in sent if sent[w] > 0)
            neg_n = sum(t.get("word_counts", {}).get(w, 0) for w in sent if sent[w] < 0)
            epi_n = sum(t.get("word_counts", {}).get(w, 0) for w in epi if w in words)
            deo_n = sum(t.get("word_counts", {}).get(w, 0) for w in deo if w in words)
            # multi-word stance markers need substring search
            agr_n = sum(text_l.count(m) for m in agr)
            dis_n = sum(text_l.count(m) for m in dis)
            if want_sent == "positive-lean" and not pos_n > neg_n:
                continue
            if want_sent == "negative-lean" and not neg_n > pos_n:
                continue
            if epi_n < want_epi or deo_n < want_deo:
                continue
            if want_stance == "agree-lean" and not agr_n >= dis_n:
                continue
            if want_stance == "disagree-lean" and not dis_n > agr_n:
                continue
            results.append({"Title": t["title"], "Positive": pos_n, "Negative": neg_n, "Epistemic": epi_n,
                            "Deontic": deo_n, "Agree": agr_n, "Disagree": dis_n})
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        st.caption("Counts come from the stored lexicon; edit it in the Dictionaries tab to refine the filter.")

# ============================================================
# TAB 9: STATISTICS & METADATA
# ============================================================
with tab9:
    st.markdown("<h3 class='subheader'>Corpus statistics</h3>", unsafe_allow_html=True)
    meta_fields = cached_metadata_fields()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Tokens per text")
        if transcripts:
            lengths = sorted([(t["title"], sum(int(v) for v in (t.get("word_counts") or {}).values())) for t in transcripts], key=lambda x: x[1])
            len_df = pd.DataFrame(lengths, columns=["text", "tokens"])
            st.plotly_chart(px.barh(len_df, x="tokens", y="text", title="Tokens per text"), use_container_width=True)
    with c2:
        st.markdown("#### Vocabulary richness")
        lm = corpus_word_freq(corpus_id, "lemma")
        n_tok = sum(lm.values())
        if n_tok:
            st.metric("Tokens", f"{n_tok:,}")
            st.metric("Types", f"{len(lm):,}")
            st.metric("TTR", f"{len(lm) / n_tok:.4f}")
            st.metric("Guiraud index", f"{len(lm) / math.sqrt(n_tok):.2f}")

    st.divider()
    st.markdown("#### Metadata coverage")
    if transcripts:
        cov_rows = []
        for f in meta_fields:
            filled = sum(1 for t in transcripts if t.get(f["field_name"]) not in (None, ""))
            cov_rows.append({"Field": f["label"], "Mandatory": "yes" if f["mandatory"] else "no", "Filled": f"{filled}/{len(transcripts)}"})
        st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
        df_all = pd.DataFrame([{f["field_name"]: t.get(f["field_name"]) for f in meta_fields} | {"title": t["title"]} for t in transcripts])
        st.download_button("Download metadata table (CSV)", df_all.to_csv(index=False), file_name="metadata.csv", mime="text/csv")

    st.divider()
    st.markdown("#### POS distribution")
    pos_total = Counter()
    for t in transcripts:
        pos_total.update({k: int(v) for k, v in (t.get("pos_counts") or {}).items()})
    if pos_total:
        pos_df = pd.DataFrame(pos_total.most_common(), columns=["POS", "Count"])
        st.plotly_chart(px.pie(pos_df.head(12), values="Count", names="POS", title="POS distribution"), use_container_width=True)

st.divider()
st.caption("Linguistic Corpus Engine v3.0 | Streamlit + spaCy + Supabase. Data persists in the cloud; anyone with the app link can browse, search and analyze. Editing/deleting is available to all viewers; use Supabase RLS if you need to restrict this.")
