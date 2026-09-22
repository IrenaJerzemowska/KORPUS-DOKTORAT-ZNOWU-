import streamlit as st
import pandas as pd
import re
import math
import plotly.express as px
import nltk
from supabase import create_client, Client

# --- 1. CONFIGURATION & DATABASE CONNECTION ---
st.set_page_config(page_title="Linguistic Corpus Engine", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

setup_nltk()

try:
    supabase = init_supabase()
except Exception as e:
    st.error(f"Initialization Error: {e}")

# --- 2. TEXT CLEANER AND NORMALIZER ---
def clean_and_normalize(text: str) -> str:
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\(\d+(\.\d+)?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_tokenize_and_pos(text: str, lang: str):
    words = re.findall(r'\b\w+\b', text.lower())
    # Szybki algorytm rozpoznawania części mowy bez ciężkich bibliotek
    tokens = []
    for i, w in enumerate(words):
        pos = "NOUN"
        if w.endswith(('ać', 'ić', 'yć', 'ing', 'ed')):
            pos = "VERB"
        elif w.endswith(('ny', 'wy', 'ki', 'ful', 'ive')):
            pos = "ADJ"
        elif w.endswith(('nie', 'wo', 'ly')):
            pos = "ADV"
        
        tokens.append({
            "token_index": i,
            "word": w,
            "lemma": w,
            "pos": pos,
            "timestamp_start": 0.0
        })
    return tokens

# --- 3. HELPER FUNCTIONS ---
def get_corpora():
    res = supabase.table("corpora").select("*").execute()
    return pd.DataFrame(res.data)

def save_transcript(corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text):
    clean_txt = clean_and_normalize(raw_text)
    res = supabase.table("transcriptions").insert({
        "corpus_id": corpus_id, "title": title, "language": lang,
        "video_url": video_url, "publication_date": str(pub_date),
        "speaker_gender": gender, "speaker_role": role,
        "register_type": reg, "raw_text": raw_text, "clean_text": clean_txt
    }).execute()
    
    t_id = res.data[0]['id']
    token_records = simple_tokenize_and_pos(clean_txt, lang)
    
    for tok in token_records:
        tok["transcript_id"] = t_id
        tok["corpus_id"] = corpus_id

    for idx in range(0, len(token_records), 500):
        supabase.table("tokens").insert(token_records[idx:idx+500]).execute()

# --- 4. NAVIGATION BAR ---
st.title("🔬 Advanced Multilingual Corpus Engine")

sidebar = st.sidebar
selected_corpus_id = None

corpora_df = get_corpora()
if not corpora_df.empty:
    corpus_choice = sidebar.selectbox("Select Active Corpus", corpora_df['name'].tolist())
    selected_corpus_id = corpora_df[corpora_df['name'] == corpus_choice]['id'].values[0]

menu = sidebar.radio("Modules", [
    "Dashboard & Upload", 
    "KWIC Concordancer & Video Sync", 
    "Regex & POS Search",
    "Keyness & Anglicism Tracker",
    "Time Series & Neologisms"
])

# --- 5. MODULES ---
if menu == "Dashboard & Upload":
    st.header("Corpus Management & Data Normalizer")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Transcriptions")
        with st.form("upload_form"):
            title = st.text_input("Transcript Title")
            lang = st.selectbox("Language", ["pl", "en"])
            video_url = st.text_input("YouTube Video URL")
            pub_date = st.date_input("Publication Date")
            gender = st.selectbox("Speaker Gender", ["Female", "Male", "Multiple", "Unknown"])
            role = st.text_input("Speaker Role")
            reg = st.selectbox("Register", ["Informal", "Formal", "Academic", "Media"])
            raw_text = st.text_area("Raw Video Transcript")
            
            submitted = st.form_submit_button("Clean, Process, & Store")
            if submitted and raw_text and selected_corpus_id:
                save_transcript(selected_corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text)
                st.success("Transcript cleaned, parsed, and persistently saved!")

    with col2:
        st.subheader("Corpus Overview & Data Editor")
        if selected_corpus_id:
            data = supabase.table("transcriptions").select("id, title, language, publication_date, register_type").eq("corpus_id", selected_corpus_id).execute()
            df = pd.DataFrame(data.data)
            st.dataframe(df)
            
            st.subheader("Delete Transcript")
            delete_title = st.selectbox("Select Item to Remove", df['title'].tolist() if not df.empty else [])
            if st.button("Permanently Delete Selected"):
                t_id = df[df['title'] == delete_title]['id'].values[0]
                supabase.table("transcriptions").delete().eq("id", t_id).execute()
                st.warning(f"Deleted '{delete_title}'. Memory updated!")
                st.rerun()

elif menu == "KWIC Concordancer & Video Sync":
    st.header("Key Word In Context (KWIC) & Video Sync")
    
    query = st.text_input("Search Word / Lemma", "polska")
    window = st.slider("Context Window (Words)", 3, 10, 5)
    
    if query and selected_corpus_id:
        res = supabase.table("tokens").select("transcript_id, token_index, word").eq("corpus_id", selected_corpus_id).ilike("lemma", query).execute()
        hits = res.data
        
        st.write(f"Total Hits: **{len(hits)}**")
        
        kwic_data = []
        for h in hits[:50]:
            t_id = h['transcript_id']
            idx = h['token_index']
            
            context_res = supabase.table("tokens").select("word, token_index").eq("transcript_id", t_id).gte("token_index", idx - window).lte("token_index", idx + window).order("token_index").execute()
            
            left_env = " ".join([t['word'] for t in context_res.data if t['token_index'] < idx])
            node = h['word']
            right_env = " ".join([t['word'] for t in context_res.data if t['token_index'] > idx])
            
            meta = supabase.table("transcriptions").select("title, video_url").eq("id", t_id).single().execute().data
            
            kwic_data.append({
                "Left Context": left_env,
                "NODE": node,
                "Right Context": right_env,
                "Source Video": meta['title'],
                "URL": meta['video_url']
            })
            
        kwic_df = pd.DataFrame(kwic_data)
        st.dataframe(kwic_df, use_container_width=True)
        
        if not kwic_df.empty:
            st.subheader("Direct Timestamp Video Player")
            selected_video = st.selectbox("Select Match to Play Video", kwic_df["Source Video"].unique())
            v_url = kwic_df[kwic_df["Source Video"] == selected_video]["URL"].values[0]
            if v_url:
                ts_sec = st.number_input("Timestamp (Seconds)", value=0)
                if "youtube.com" in v_url or "youtu.be" in v_url:
                    v_url_ts = f"{v_url}&t={ts_sec}s" if "?" in v_url else f"{v_url}?t={ts_sec}s"
                    st.video(v_url_ts)
                else:
                    st.video(v_url)

elif menu == "Regex & POS Search":
    st.header("Regex & POS Sequence Syntax Search")
    
    c1, c2 = st.columns(2)
    with c1:
        regex_pattern = st.text_input("Regex Search Pattern", r"\b[A-Za-z]+acja\b")
    with c2:
        pos_pattern = st.selectbox("POS Filter", ["ANY", "NOUN", "VERB", "ADJ", "ADV"])
        
    if st.button("Execute Pattern Search"):
        tokens_res = supabase.table("tokens").select("word, lemma, pos").eq("corpus_id", selected_corpus_id).execute()
        tdf = pd.DataFrame(tokens_res.data)
        
        if not tdf.empty:
            if pos_pattern != "ANY":
                tdf = tdf[tdf['pos'] == pos_pattern]
            
            filtered = tdf[tdf['word'].str.contains(regex_pattern, regex=True, na=False)]
            st.write(f"Matches Found: **{len(filtered)}**")
            st.dataframe(filtered[['word', 'lemma', 'pos']].value_counts().reset_index(name='Frequency'))

elif menu == "Keyness & Anglicism Tracker":
    st.header("Anglicism & Loanword Spotter")
    
    anglicism_suffixes = ['ing', 'yom', 'er', 'ster', 'ed', 'em']
    
    if st.button("Spot Loanwords"):
        t_data = supabase.table("tokens").select("lemma").eq("corpus_id", selected_corpus_id).execute()
        target_df = pd.DataFrame(t_data.data)
        
        if not target_df.empty:
            loanwords = target_df[target_df['lemma'].str.contains(r'(' + '|'.join(anglicism_suffixes) + r')$', regex=True, na=False)]
            st.subheader("Potential Anglicism Candidates")
            st.dataframe(loanwords['lemma'].value_counts().reset_index(name='Occurrences').head(20))

elif menu == "Time Series & Neologisms":
    st.header("Frequency Trends")
    target_word = st.text_input("Track Word over time:", "polska")
    
    if target_word and selected_corpus_id:
        res = supabase.table("tokens").select("lemma, transcriptions(publication_date)").eq("corpus_id", selected_corpus_id).eq("lemma", target_word.lower()).execute()
        
        data = [{"date": row['transcriptions']['publication_date']} for row in res.data if row['transcriptions']]
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df_grouped = df.groupby(df['date'].dt.to_period("M")).size().reset_index(name='Frequency')
            df_grouped['date'] = df_grouped['date'].astype(str)
            fig = px.line(df_grouped, x='date', y='Frequency', title=f"Temporal Trend for '{target_word}'")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matching historical occurrences found.")
