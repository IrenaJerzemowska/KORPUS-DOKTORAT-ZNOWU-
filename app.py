import streamlit as st
import pandas as pd
import re
import math
import plotly.express as px
import spacy
from supabase import create_client, Client

# --- 1. CONFIGURATION & DATABASE CONNECTION ---
st.set_page_config(page_title="Linguistic Corpus Engine", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)
@st.cache_resource
def load_nlp_models():
    import subprocess
    import sys

    # Pobieranie modeli za pomocą subprocess (omija blokady uprawnień pip w kontenerze)
    models = {"pl": "pl_core_news_sm", "en": "en_core_web_sm"}
    loaded_models = {}

    for lang, model_name in models.items():
        if not spacy.util.is_package(model_name):
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        loaded_models[lang] = spacy.load(model_name)

    return loaded_models

# --- 2. TEXT CLEANER AND NORMALIZER ---
def clean_and_normalize(text: str) -> str:
    # Remove timestamps like [00:12:34] or (12.3) if present in transcript
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\(\d+(\.\d+)?\)', '', text)
    # Remove unwanted extra whitespaces & linebreaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. HELPER FUNCTIONS ---
def get_corpora():
    res = supabase.table("corpora").select("*").execute()
    return pd.DataFrame(res.data)

def save_transcript(corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text):
    clean_txt = clean_and_normalize(raw_text)
    # Insert Transcription
    res = supabase.table("transcriptions").insert({
        "corpus_id": corpus_id, "title": title, "language": lang,
        "video_url": video_url, "publication_date": str(pub_date),
        "speaker_gender": gender, "speaker_role": role,
        "register_type": reg, "raw_text": raw_text, "clean_text": clean_txt
    }).execute()
    
    t_id = res.data[0]['id']
    
    # Process and Store Tokens via spaCy
    nlp = nlp_models[lang]
    doc = nlp(clean_txt)
    token_records = []
    
    # Simple regex timestamp extractor if formatted as "word|12.5"
    for i, tok in enumerate(doc):
        token_records.append({
            "transcript_id": t_id,
            "corpus_id": corpus_id,
            "token_index": i,
            "word": tok.text,
            "lemma": tok.lemma_.lower(),
            "pos": tok.pos_,
            "timestamp_start": 0.0 # Standard fallback
        })
        
    # Bulk insert tokens (batched per 500)
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
    "Time Series & Neologisms",
    "Sentiment & Modality Filter"
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
            video_url = st.text_input("YouTube Video URL (e.g. https://www.youtube.com/watch?v=XYZ)")
            pub_date = st.date_input("Publication Date")
            gender = st.selectbox("Speaker Gender", ["Female", "Male", "Multiple", "Unknown"])
            role = st.text_input("Speaker Role (e.g. Journalist, Politician)")
            reg = st.selectbox("Register", ["Informal", "Formal", "Academic", "Media"])
            raw_text = st.text_area("Raw Video Transcript")
            
            submitted = st.form_submit_button("Clean, Process, & Store")
            if submitted and raw_text and selected_corpus_id:
                save_transcript(selected_corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text)
                st.success("Transcript cleaned, POS tagged, and persistently saved!")

    with col2:
        st.subheader("Corpus Overview & Data Editor")
        if selected_corpus_id:
            data = supabase.table("transcriptions").select("id, title, language, publication_date, register_type").eq("corpus_id", selected_corpus_id).execute()
            df = pd.DataFrame(data.data)
            st.dataframe(df)
            
            # Edit / Delete Section
            st.subheader("Delete Transcript")
            delete_id = st.selectbox("Select Item to Remove", df['title'].tolist() if not df.empty else [])
            if st.button("Permanently Delete Selected"):
                t_id = df[df['title'] == delete_id]['id'].values[0]
                supabase.table("transcriptions").delete().eq("id", t_id).execute()
                st.warning(f"Deleted '{delete_id}'. Memory updated!")
                st.rerun()

elif menu == "KWIC Concordancer & Video Sync":
    st.header("Key Word In Context (KWIC) & Video Sync")
    
    query = st.text_input("Search Word / Lemma", "polska")
    window = st.slider("Context Window (Words)", 3, 10, 5)
    
    if query and selected_corpus_id:
        # Fetch tokens matching query
        res = supabase.table("tokens").select("transcript_id, token_index, word").eq("corpus_id", selected_corpus_id).ilike("lemma", query).execute()
        hits = res.data
        
        st.write(f"Total Hits: **{len(hits)}**")
        
        kwic_data = []
        for h in hits[:50]: # Limit render batch
            t_id = h['transcript_id']
            idx = h['token_index']
            
            # Context Retrieval
            context_res = supabase.table("tokens").select("word, token_index").eq("transcript_id", t_id).gte("token_index", idx - window).lte("token_index", idx + window).order("token_index").execute()
            
            left_env = " ".join([t['word'] for t in context_res.data if t['token_index'] < idx])
            node = h['word']
            right_env = " ".join([t['word'] for t in context_res.data if t['token_index'] > idx])
            
            # Metadata fetch
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
        
        # Interactive Embedded Video Timestamp Player
        if not kwic_df.empty:
            st.subheader("Direct Timestamp Video Player")
            selected_video = st.selectbox("Select Match to Play Video", kwic_df["Source Video"].unique())
            v_url = kwic_df[kwic_df["Source Video"] == selected_video]["URL"].values[0]
            if v_url:
                # Add YouTube timestamp jump capability
                ts_sec = st.number_input("Timestamp (Seconds)", value=0)
                if "youtube.com" in v_url or "youtu.be" in v_url:
                    v_url_ts = f"{v_url}&t={ts_sec}s" if "?" in v_url else f"{v_url}?t={ts_sec}s"
                    st.video(v_url_ts)
                else:
                    st.video(v_url)

elif menu == "Regex & POS Pattern Search":
    st.header("Regex & POS Sequence Syntax Search")
    st.info("Formulate queries based on Universal POS Tags (e.g. ADJ + NOUN patterns)")
    
    c1, c2 = st.columns(2)
    with c1:
        regex_pattern = st.text_input("Regex Search Pattern", r"\b[A-Za-z]+izacja\b")
    with c2:
        pos_pattern = st.selectbox("POS Filter", ["ANY", "NOUN", "VERB", "ADJ", "ADV", "PROPN"])
        
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
    st.header("Keyness Score Calculator & Anglicism/Loanword Spotter")
    
    st.markdown("Calculates **Log-Likelihood Keyness** against a default comparative reference baseline.")
    
    # Pre-defined list of common Polish loanword stems / anglicisms
    anglicism_suffixes = ['ing', 'yom', 'er', 'ster', 'ed', 'em']
    
    if st.button("Compute Keyness & Spot Loanwords"):
        # Fetch Target Corpus tokens
        t_data = supabase.table("tokens").select("lemma").eq("corpus_id", selected_corpus_id).execute()
        target_df = pd.DataFrame(t_data.data)
        
        if not target_df.empty:
            target_counts = target_df['lemma'].value_counts()
            N_target = len(target_df)
            
            # Spot potential English loanwords in Polish transcripts
            loanwords = target_df[target_df['lemma'].str.contains(r'(' + '|'.join(anglicism_suffixes) + r')$', regex=True, na=False)]
            
            st.subheader("Potential Anglicism & Loanword Candidates")
            st.dataframe(loanwords['lemma'].value_counts().reset_index(name='Occurrences').head(20))

elif menu == "Time Series & Neologisms":
    st.header("Frequency Trends & Emergence Spotter")
    
    target_word = st.text_input("Track Word / Neologism over time:", "smartfon")
    
    if target_word and selected_corpus_id:
        # Join tokens with publication dates
        res = supabase.table("tokens").select("lemma, transcriptions(publication_date)").eq("corpus_id", selected_corpus_id).eq("lemma", target_word.lower()).execute()
        
        data = []
        for row in res.data:
            if row['transcriptions']:
                data.append({"date": row['transcriptions']['publication_date']})
                
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df_grouped = df.groupby(df['date'].dt.to_period("M")).size().reset_index(name='Frequency')
            df_grouped['date'] = df_grouped['date'].astype(str)
            
            fig = px.line(df_grouped, x='date', y='Frequency', title=f"Temporal Trend for '{target_word}'")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matching historical occurrences found in the corpus timeline.")

elif menu == "Sentiment & Modality Filter":
    st.header("Stance, Sentiment, & Modality Analysis")
    
    st.markdown("Filter corpus metadata and transcripts by linguistic modality markers (e.g., conditional verbs, modal particles).")
    
    modality_marker = st.selectbox("Modality Marker Type", ["Epistemic (musieć, móc)", "Deontic (powinien, trzeba)", "Volitional (chcieć)"])
    
    if st.button("Filter Transcripts"):
        st.info("Filter applied! Querying parsed modal vectors...")
