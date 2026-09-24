import streamlit as st
import pandas as pd
import re
import plotly.express as px
import spacy
from supabase import create_client, Client

# --- 1. CONFIGURATION & DATABASE CONNECTION ---
st.set_page_config(page_title="Linguistic Corpus Engine", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        
        if not url or not key:
            st.error("❌ Missing Supabase secrets. Add SUPABASE_URL and SUPABASE_KEY to your Streamlit Cloud secrets.")
            st.stop()
        
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase Connection Error: {e}")
        st.stop()

@st.cache_resource
def load_spacy_model(lang_code: str):
    """Load spaCy model based on language code"""
    model_map = {
        "pl": "pl_core_news_sm",
        "en": "en_core_web_sm",
        "zh": "zh_core_web_sm",
    }
    
    model_name = model_map.get(lang_code, "en_core_web_sm")
    
    try:
        # Try to load the model
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        st.warning(f"⚠️ Model '{model_name}' not found. Downloading...")
        try:
            import subprocess
            subprocess.run([f"python -m spacy download {model_name}"], shell=True, check=True)
            nlp = spacy.load(model_name)
            return nlp
        except Exception as e:
            st.error(f"❌ Failed to load spaCy model: {e}")
            st.stop()

supabase = init_supabase()

# --- 2. TEXT CLEANER AND NORMALIZER ---
def clean_and_normalize(text: str) -> str:
    """Remove timestamps and normalize whitespace"""
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\(\d+(\.\d+)?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pos_spacy(text: str, lang: str):
    """Tokenize and assign POS tags using spaCy"""
    nlp = load_spacy_model(lang)
    doc = nlp(text)
    
    tokens = []
    for i, token in enumerate(doc):
        tokens.append({
            "token_index": i,
            "word": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "timestamp_start": 0.0
        })
    
    return tokens

# --- 3. HELPER FUNCTIONS ---
def get_corpora():
    """Fetch all available corpora"""
    try:
        res = supabase.table("corpora").select("*").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching corpora: {e}")
        return pd.DataFrame()

def save_transcript(corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text):
    """Clean, tokenize with spaCy, and save transcript to Supabase"""
    try:
        clean_txt = clean_and_normalize(raw_text)
        
        # Insert transcript
        res = supabase.table("transcriptions").insert({
            "corpus_id": corpus_id,
            "title": title,
            "language": lang,
            "video_url": video_url,
            "publication_date": str(pub_date),
            "speaker_gender": gender,
            "speaker_role": role,
            "register_type": reg,
            "raw_text": raw_text,
            "clean_text": clean_txt
        }).execute()
        
        if not res.data:
            st.error("Failed to insert transcript")
            return
        
        t_id = res.data[0]['id']
        
        # Tokenize with spaCy
        with st.spinner(f"🔬 Tokenizing with spaCy ({lang})..."):
            token_records = tokenize_and_pos_spacy(clean_txt, lang)
        
        # Add transcript_id and corpus_id to tokens
        for tok in token_records:
            tok["transcript_id"] = t_id
            tok["corpus_id"] = corpus_id
        
        # Batch insert tokens (Supabase has limits)
        for idx in range(0, len(token_records), 500):
            batch = token_records[idx:idx+500]
            supabase.table("tokens").insert(batch).execute()
        
        st.success(f"✅ Transcript saved! ({len(token_records)} tokens)")
        
    except Exception as e:
        st.error(f"Error saving transcript: {e}")

# --- 4. NAVIGATION BAR ---
st.title("🔬 Advanced Multilingual Corpus Engine")

sidebar = st.sidebar
selected_corpus_id = None

corpora_df = get_corpora()

if not corpora_df.empty:
    corpus_choice = sidebar.selectbox("Select Active Corpus", corpora_df['name'].tolist())
    selected_corpus_id = corpora_df[corpora_df['name'] == corpus_choice]['id'].values[0]
else:
    st.warning("⚠️ No corpora found. Create one in your database first.")
    st.stop()

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
            title = st.text_input("Transcript Title", value="")
            lang = st.selectbox("Language", ["pl", "en", "zh"])
            video_url = st.text_input("YouTube Video URL", value="")
            pub_date = st.date_input("Publication Date")
            gender = st.selectbox("Speaker Gender", ["Female", "Male", "Multiple", "Unknown"])
            role = st.text_input("Speaker Role", value="")
            reg = st.selectbox("Register", ["Informal", "Formal", "Academic", "Media"])
            raw_text = st.text_area("Raw Video Transcript", height=200)
            
            submitted = st.form_submit_button("Clean, Process, & Store")
            
            if submitted:
                if not raw_text.strip():
                    st.error("❌ Please enter transcript text")
                elif not title.strip():
                    st.error("❌ Please enter a title")
                else:
                    save_transcript(selected_corpus_id, title, lang, video_url, pub_date, gender, role, reg, raw_text)

    with col2:
        st.subheader("Corpus Overview")
        if selected_corpus_id:
            try:
                data = supabase.table("transcriptions").select(
                    "id, title, language, publication_date, register_type"
                ).eq("corpus_id", selected_corpus_id).execute()
                
                df = pd.DataFrame(data.data) if data.data else pd.DataFrame()
                
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    st.subheader("Delete Transcript")
                    delete_title = st.selectbox("Select Item to Remove", df['title'].tolist())
                    
                    if st.button("Permanently Delete Selected"):
                        try:
                            t_id = df[df['title'] == delete_title]['id'].values[0]
                            supabase.table("transcriptions").delete().eq("id", t_id).execute()
                            st.warning(f"🗑️ Deleted '{delete_title}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete error: {e}")
                else:
                    st.info("No transcripts yet. Upload one above!")
                    
            except Exception as e:
                st.error(f"Error loading transcripts: {e}")

elif menu == "KWIC Concordancer & Video Sync":
    st.header("Key Word In Context (KWIC) & Video Sync")
    
    query = st.text_input("Search Word / Lemma", "polska")
    window = st.slider("Context Window (Words)", 3, 10, 5)
    
    if query and selected_corpus_id:
        try:
            res = supabase.table("tokens").select(
                "transcript_id, token_index, word"
            ).eq("corpus_id", selected_corpus_id).ilike("lemma", f"%{query}%").execute()
            
            hits = res.data if res.data else []
            st.write(f"**Total Hits: {len(hits)}**")
            
            if hits:
                kwic_data = []
                
                for h in hits[:50]:  # Limit to 50 for performance
                    t_id = h['transcript_id']
                    idx = h['token_index']
                    
                    try:
                        context_res = supabase.table("tokens").select(
                            "word, token_index"
                        ).eq("transcript_id", t_id).gte(
                            "token_index", idx - window
                        ).lte(
                            "token_index", idx + window
                        ).order("token_index").execute()
                        
                        context_data = context_res.data if context_res.data else []
                        
                        left_env = " ".join([t['word'] for t in context_data if t['token_index'] < idx])
                        node = h['word']
                        right_env = " ".join([t['word'] for t in context_data if t['token_index'] > idx])
                        
                        meta_res = supabase.table("transcriptions").select(
                            "title, video_url"
                        ).eq("id", t_id).execute()
                        
                        if meta_res.data:
                            meta = meta_res.data[0]
                            kwic_data.append({
                                "Left Context": left_env,
                                "NODE": node,
                                "Right Context": right_env,
                                "Source Video": meta['title'],
                                "URL": meta['video_url']
                            })
                    except Exception as e:
                        st.warning(f"Error processing hit: {e}")
                        continue
                
                if kwic_data:
                    kwic_df = pd.DataFrame(kwic_data)
                    st.dataframe(kwic_df, use_container_width=True)
                    
                    st.subheader("📹 Video Reference")
                    if not kwic_df.empty:
                        selected_video = st.selectbox("Select Match Source", kwic_df["Source Video"].unique())
                        v_url = kwic_df[kwic_df["Source Video"] == selected_video]["URL"].values[0]
                        
                        if v_url:
                            st.markdown(f"**Source:** [{selected_video}]({v_url})")
                            st.info("Note: Click the link above to watch the video context in YouTube")
                else:
                    st.info("No results found")
            else:
                st.info("No matches found")
                
        except Exception as e:
            st.error(f"Search error: {e}")

elif menu == "Regex & POS Search":
    st.header("Regex & POS Sequence Syntax Search")
    
    c1, c2 = st.columns(2)
    with c1:
        regex_pattern = st.text_input("Regex Search Pattern (word)", r"\b[A-Za-z]+acja\b")
    with c2:
        pos_pattern = st.selectbox("POS Filter", ["ANY", "NOUN", "VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ", "PROPN"])
    
    if st.button("Execute Pattern Search"):
        try:
            tokens_res = supabase.table("tokens").select(
                "word, lemma, pos, tag"
            ).eq("corpus_id", selected_corpus_id).execute()
            
            tdf = pd.DataFrame(tokens_res.data) if tokens_res.data else pd.DataFrame()
            
            if not tdf.empty:
                if pos_pattern != "ANY":
                    tdf = tdf[tdf['pos'] == pos_pattern]
                
                try:
                    filtered = tdf[tdf['word'].str.contains(regex_pattern, regex=True, na=False)]
                    st.write(f"**Matches Found: {len(filtered)}**")
                    
                    if not filtered.empty:
                        freq_table = filtered[['word', 'lemma', 'pos']].value_counts().reset_index(name='Frequency')
                        st.dataframe(freq_table, use_container_width=True)
                    else:
                        st.info("No matches found for this pattern")
                except re.error as e:
                    st.error(f"Invalid regex pattern: {e}")
            else:
                st.info("No tokens found in corpus")
                
        except Exception as e:
            st.error(f"Search error: {e}")

elif menu == "Keyness & Anglicism Tracker":
    st.header("Anglicism & Loanword Spotter")
    
    anglicism_suffixes = ['ing', 'yom', 'er', 'ster', 'ed', 'em', 'ism', 'ity']
    
    if st.button("Spot Loanwords"):
        try:
            t_data = supabase.table("tokens").select("lemma, pos").eq("corpus_id", selected_corpus_id).execute()
            target_df = pd.DataFrame(t_data.data) if t_data.data else pd.DataFrame()
            
            if not target_df.empty:
                pattern = r'(' + '|'.join(anglicism_suffixes) + r')$'
                loanwords = target_df[target_df['lemma'].str.contains(pattern, regex=True, na=False)]
                
                st.subheader("🌍 Potential Anglicism Candidates")
                if not loanwords.empty:
                    freq_table = loanwords['lemma'].value_counts().reset_index(name='Occurrences').head(20)
                    st.dataframe(freq_table, use_container_width=True)
                else:
                    st.info("No loanwords found")
            else:
                st.info("No tokens in corpus")
                
        except Exception as e:
            st.error(f"Analysis error: {e}")

elif menu == "Time Series & Neologisms":
    st.header("Frequency Trends Over Time")
    target_word = st.text_input("Track Word over time:", "polska")
    
    if target_word and selected_corpus_id:
        try:
            res = supabase.table("tokens").select(
                "lemma, transcriptions(publication_date)"
            ).eq("corpus_id", selected_corpus_id).ilike("lemma", f"%{target_word}%").execute()
            
            data = [
                {"date": row['transcriptions']['publication_date']} 
                for row in (res.data or []) 
                if row.get('transcriptions')
            ]
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df_grouped = df.groupby(df['date'].dt.to_period("M")).size().reset_index(name='Frequency')
                df_grouped['date'] = df_grouped['date'].astype(str)
                
                fig = px.line(
                    df_grouped, 
                    x='date', 
                    y='Frequency', 
                    title=f"Temporal Trend for '{target_word}'",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No historical data found for '{target_word}'")
                
        except Exception as e:
            st.error(f"Analysis error: {e}")
