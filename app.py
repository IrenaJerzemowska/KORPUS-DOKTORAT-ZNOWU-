import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import spacy
from supabase import create_client, Client
from datetime import datetime
from collections import Counter
import io

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Linguistic Corpus Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ---
st.markdown("""
<style>
    .main-header { font-size: 2.5em; font-weight: bold; color: #1f77b4; margin-bottom: 0.5em; }
    .subheader { font-size: 1.3em; font-weight: bold; color: #333; margin-top: 1em; margin-bottom: 0.5em; }
    .stat-box { 
        background-color: #f0f2f6; 
        padding: 1.5em; 
        border-radius: 0.5em; 
        margin: 0.5em 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box { background-color: #d4edda; padding: 1em; border-radius: 0.5em; color: #155724; }
    .error-box { background-color: #f8d7da; padding: 1em; border-radius: 0.5em; color: #721c24; }
    .kwic-node { font-weight: bold; background-color: #fff3cd; padding: 0.2em 0.4em; border-radius: 0.3em; }
</style>
""", unsafe_allow_html=True)

# --- DATABASE CONNECTION ---
@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        
        if not url or not key:
            st.error("❌ Missing Supabase secrets")
            st.stop()
        
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        st.stop()

@st.cache_resource
def load_spacy_model(lang_code: str):
    """Load spaCy model with automatic download"""
    model_map = {
        "pl": "pl_core_news_sm",
        "en": "en_core_web_sm",
        "zh": "zh_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
    }
    
    model_name = model_map.get(lang_code, "en_core_web_sm")
    
    try:
        return spacy.load(model_name)
    except OSError:
        st.info(f"⏳ Downloading {model_name}...")
        import subprocess
        subprocess.run([f"python -m spacy download {model_name}"], shell=True, check=True)
        return spacy.load(model_name)

supabase = init_supabase()

# --- HELPER FUNCTIONS ---
def clean_and_normalize(text: str) -> str:
    """Remove timestamps and normalize text"""
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\(\d+(\.\d+)?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_spacy(text: str, lang: str):
    """Tokenize and tag with spaCy"""
    nlp = load_spacy_model(lang)
    doc = nlp(text)
    
    tokens = []
    for i, token in enumerate(doc):
        tokens.append({
            "token_index": i,
            "word": token.text.lower(),
            "lemma": token.lemma_.lower(),
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "timestamp_start": 0.0
        })
    
    return tokens

# --- CORPUS MANAGEMENT ---
def get_all_corpora():
    """Fetch all corpora"""
    try:
        res = supabase.table("corpora").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching corpora: {e}")
        return pd.DataFrame()

def create_corpus(name: str, description: str, language: str):
    """Create new corpus"""
    try:
        res = supabase.table("corpora").insert({
            "name": name,
            "description": description,
            "language": language,
            "created_at": datetime.now().isoformat(),
            "token_count": 0
        }).execute()
        
        if res.data:
            st.success(f"✅ Corpus '{name}' created!")
            st.rerun()
        else:
            st.error("Failed to create corpus")
    except Exception as e:
        st.error(f"Error creating corpus: {e}")

def delete_corpus(corpus_id: int):
    """Delete entire corpus with all transcriptions and tokens"""
    try:
        # Delete tokens first
        supabase.table("tokens").delete().eq("corpus_id", corpus_id).execute()
        # Delete transcriptions
        supabase.table("transcriptions").delete().eq("corpus_id", corpus_id).execute()
        # Delete corpus
        supabase.table("corpora").delete().eq("id", corpus_id).execute()
        
        st.success("✅ Corpus deleted!")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting corpus: {e}")

def get_corpus_stats(corpus_id: int):
    """Get statistics for a corpus"""
    try:
        # Get transcription count
        trans_res = supabase.table("transcriptions").select("id", count="exact").eq("corpus_id", corpus_id).execute()
        trans_count = trans_res.count if hasattr(trans_res, 'count') else len(trans_res.data or [])
        
        # Get token count
        token_res = supabase.table("tokens").select("id", count="exact").eq("corpus_id", corpus_id).execute()
        token_count = token_res.count if hasattr(token_res, 'count') else len(token_res.data or [])
        
        # Get unique lemmas
        lemma_res = supabase.table("tokens").select("lemma").eq("corpus_id", corpus_id).execute()
        lemma_count = len(set([t['lemma'] for t in (lemma_res.data or [])]))
        
        return {
            "transcriptions": trans_count,
            "tokens": token_count,
            "types": lemma_count
        }
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return {"transcriptions": 0, "tokens": 0, "types": 0}

def save_transcript(corpus_id: int, title: str, lang: str, raw_text: str, metadata: dict = None):
    """Save and process transcript"""
    try:
        clean_txt = clean_and_normalize(raw_text)
        
        # Insert transcript
        transcript_data = {
            "corpus_id": corpus_id,
            "title": title,
            "language": lang,
            "raw_text": raw_text,
            "clean_text": clean_txt,
            "created_at": datetime.now().isoformat()
        }
        
        if metadata:
            transcript_data.update(metadata)
        
        res = supabase.table("transcriptions").insert(transcript_data).execute()
        
        if not res.data:
            st.error("Failed to insert transcript")
            return
        
        t_id = res.data[0]['id']
        
        # Tokenize with spaCy
        with st.spinner(f"🔬 Processing text with spaCy..."):
            token_records = tokenize_with_spacy(clean_txt, lang)
        
        # Add IDs
        for tok in token_records:
            tok["transcript_id"] = t_id
            tok["corpus_id"] = corpus_id
        
        # Batch insert tokens
        for idx in range(0, len(token_records), 500):
            batch = token_records[idx:idx+500]
            supabase.table("tokens").insert(batch).execute()
        
        st.success(f"✅ Text saved! ({len(token_records)} tokens)")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error saving transcript: {e}")

# --- MAIN UI ---
st.markdown("<h1 class='main-header'>📚 Linguistic Corpus Engine</h1>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🔧 Corpus Management")
    
    tab_manage = st.tabs(["Corpora", "Create New"])
    
    with tab_manage[0]:
        corpora_df = get_all_corpora()
        
        if corpora_df.empty:
            st.info("No corpora yet. Create one →")
            selected_corpus_id = None
        else:
            corpus_names = corpora_df['name'].tolist()
            selected_name = st.selectbox("Active Corpus", corpus_names, key="corpus_select")
            selected_corpus_id = corpora_df[corpora_df['name'] == selected_name]['id'].values[0]
            
            st.divider()
            
            if st.button("🗑️ Delete This Corpus", type="secondary"):
                if st.checkbox("Confirm deletion"):
                    delete_corpus(selected_corpus_id)
    
    with tab_manage[1]:
        st.markdown("### Create New Corpus")
        new_name = st.text_input("Corpus Name", placeholder="e.g., 'Modern Polish News'")
        new_desc = st.text_area("Description", placeholder="What texts does this contain?", height=80)
        new_lang = st.selectbox("Language", ["pl", "en", "de", "fr", "zh"])
        
        if st.button("✨ Create Corpus", type="primary", use_container_width=True):
            if new_name:
                create_corpus(new_name, new_desc, new_lang)
            else:
                st.error("Enter a corpus name")

# --- MAIN CONTENT ---
if not st.session_state.get('selected_corpus_id') and 'selected_corpus_id' in locals():
    st.session_state['selected_corpus_id'] = selected_corpus_id

corpus_id = st.session_state.get('selected_corpus_id') or selected_corpus_id

if not corpus_id:
    st.warning("⬅️ Select or create a corpus to start")
    st.stop()

# --- CORPUS HEADER ---
corpora_df = get_all_corpora()
active_corpus = corpora_df[corpora_df['id'] == corpus_id].iloc[0] if not corpora_df.empty else None

if active_corpus is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    stats = get_corpus_stats(corpus_id)
    
    with col1:
        st.metric("📄 Texts", stats['transcriptions'])
    with col2:
        st.metric("🔤 Tokens", f"{stats['tokens']:,}")
    with col3:
        st.metric("📋 Types", f"{stats['types']:,}")
    with col4:
        if stats['tokens'] > 0:
            ttr = stats['types'] / stats['tokens']
            st.metric("Type/Token Ratio", f"{ttr:.3f}")

st.divider()

# --- TAB NAVIGATION ---
tabs = st.tabs([
    "📤 Upload & Manage",
    "🔍 KWIC Search",
    "📊 Frequency Analysis",
    "🔎 Advanced Search",
    "📈 Trends & Statistics"
])

# === TAB 1: UPLOAD & MANAGE ===
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='subheader'>📝 Add Text to Corpus</h3>", unsafe_allow_html=True)
        
        upload_method = st.radio("Choose method:", ["Paste Text", "Upload File"], horizontal=True)
        
        if upload_method == "Paste Text":
            title = st.text_input("Text Title", placeholder="e.g., 'News Article 001'")
            text_input = st.text_area("Text Content", height=250, placeholder="Paste your text here...")
            
            if st.button("💾 Add to Corpus", type="primary", use_container_width=True):
                if text_input.strip() and title:
                    save_transcript(corpus_id, title, active_corpus['language'], text_input)
                else:
                    st.error("Enter title and text")
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
            if uploaded_file:
                try:
                    if uploaded_file.type == "text/plain":
                        text_content = uploaded_file.read().decode("utf-8")
                    else:
                        try:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            text_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
                        except:
                            st.error("Install PyPDF2 for PDF support")
                            text_content = None
                    
                    if text_content:
                        file_title = uploaded_file.name.replace(".txt", "").replace(".pdf", "")
                        if st.button("💾 Add File to Corpus", type="primary", use_container_width=True):
                            save_transcript(corpus_id, file_title, active_corpus['language'], text_content)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("<h3 class='subheader'>📚 Texts in Corpus</h3>", unsafe_allow_html=True)
        
        try:
            trans_res = supabase.table("transcriptions").select(
                "id, title, created_at, language"
            ).eq("corpus_id", corpus_id).order("created_at", desc=True).execute()
            
            if trans_res.data:
                trans_df = pd.DataFrame(trans_res.data)
                trans_df['created_at'] = pd.to_datetime(trans_df['created_at']).dt.strftime("%Y-%m-%d")
                
                for _, row in trans_df.iterrows():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.caption(f"📄 **{row['title']}** ({row['created_at']})")
                    with col_b:
                        if st.button("🗑️", key=f"del_{row['id']}", help="Delete"):
                            supabase.table("tokens").delete().eq("transcript_id", row['id']).execute()
                            supabase.table("transcriptions").delete().eq("id", row['id']).execute()
                            st.success("Deleted!")
                            st.rerun()
            else:
                st.info("No texts added yet")
        except Exception as e:
            st.error(f"Error loading texts: {e}")

# === TAB 2: KWIC SEARCH ===
with tabs[1]:
    st.markdown("### 🔍 Keyword In Context")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_word = st.text_input("Search lemma or word:", placeholder="e.g., 'polska'")
    with col2:
        context_window = st.slider("Context", 2, 15, 5)
    
    if search_word and st.button("Search", type="primary"):
        try:
            hits_res = supabase.table("tokens").select(
                "transcript_id, token_index, word"
            ).eq("corpus_id", corpus_id).ilike("lemma", f"%{search_word.lower()}%").execute()
            
            hits = hits_res.data or []
            
            st.info(f"Found {len(hits)} matches")
            
            if hits:
                kwic_list = []
                
                for h in hits[:100]:
                    t_id = h['transcript_id']
                    idx = h['token_index']
                    
                    context_res = supabase.table("tokens").select(
                        "word, token_index"
                    ).eq("transcript_id", t_id).gte(
                        "token_index", idx - context_window
                    ).lte(
                        "token_index", idx + context_window
                    ).order("token_index").execute()
                    
                    context_data = context_res.data or []
                    
                    left = " ".join([t['word'] for t in context_data if t['token_index'] < idx])
                    right = " ".join([t['word'] for t in context_data if t['token_index'] > idx])
                    
                    kwic_list.append({
                        "Left": left,
                        "Keyword": h['word'].upper(),
                        "Right": right
                    })
                
                kwic_df = pd.DataFrame(kwic_list)
                st.dataframe(kwic_df, use_container_width=True, hide_index=True)
                
                # Download option
                csv = kwic_df.to_csv(index=False)
                st.download_button("⬇️ Download as CSV", csv, file_name="kwic.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"Search error: {e}")

# === TAB 3: FREQUENCY ANALYSIS ===
with tabs[2]:
    st.markdown("### 📊 Word Frequency")
    
    freq_type = st.radio("Show:", ["Lemma Frequency", "POS Distribution"], horizontal=True)
    top_n = st.slider("Top N results", 5, 100, 20)
    
    if st.button("Analyze", type="primary"):
        try:
            tokens_res = supabase.table("tokens").select("lemma, pos").eq("corpus_id", corpus_id).execute()
            
            if tokens_res.data:
                if freq_type == "Lemma Frequency":
                    lemmas = [t['lemma'] for t in tokens_res.data]
                    freq_counter = Counter(lemmas)
                    
                    freq_df = pd.DataFrame(
                        freq_counter.most_common(top_n),
                        columns=['Lemma', 'Frequency']
                    )
                    freq_df['Relative Freq (%)'] = (freq_df['Frequency'] / len(lemmas) * 100).round(2)
                    
                    # Chart
                    fig = px.bar(
                        freq_df.head(15),
                        x='Lemma',
                        y='Frequency',
                        title='Most Frequent Lemmas',
                        color='Frequency',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)
                
                else:  # POS Distribution
                    pos_list = [t['pos'] for t in tokens_res.data]
                    pos_counter = Counter(pos_list)
                    
                    pos_df = pd.DataFrame(
                        pos_counter.most_common(),
                        columns=['POS', 'Count']
                    )
                    
                    fig = px.pie(pos_df, values='Count', names='POS', title='POS Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Analysis error: {e}")

# === TAB 4: ADVANCED SEARCH ===
with tabs[3]:
    st.markdown("### 🔎 Advanced Search")
    
    search_by = st.radio("Search by:", ["Regex", "POS Tag", "Lemma + POS"], horizontal=True)
    
    if search_by == "Regex":
        pattern = st.text_input("Regex pattern:", r"\b[A-Za-z]+acja\b")
        
        if st.button("Search Regex", type="primary"):
            try:
                tokens_res = supabase.table("tokens").select("word, lemma, pos").eq("corpus_id", corpus_id).execute()
                
                if tokens_res.data:
                    tdf = pd.DataFrame(tokens_res.data)
                    filtered = tdf[tdf['word'].str.contains(pattern, regex=True, na=False, case=False)]
                    
                    if not filtered.empty:
                        freq_df = filtered['lemma'].value_counts().head(20).reset_index()
                        freq_df.columns = ['Lemma', 'Frequency']
                        
                        st.info(f"Found {len(filtered)} matches")
                        st.dataframe(freq_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No matches found")
            except re.error as e:
                st.error(f"Invalid regex: {e}")
    
    elif search_by == "POS Tag":
        pos_tag = st.selectbox("Select POS:", [
            "NOUN", "VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ", "PROPN", "PRON", "DET"
        ])
        
        if st.button("Search POS", type="primary"):
            try:
                tokens_res = supabase.table("tokens").select("word, lemma, pos").eq("corpus_id", corpus_id).eq("pos", pos_tag).execute()
                
                if tokens_res.data:
                    freq_df = pd.DataFrame(tokens_res.data)['lemma'].value_counts().head(30).reset_index()
                    freq_df.columns = ['Lemma', 'Frequency']
                    
                    st.info(f"Found {len(tokens_res.data)} tokens with POS={pos_tag}")
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Search error: {e}")
    
    else:  # Lemma + POS
        lemma = st.text_input("Lemma:")
        pos = st.selectbox("POS:", ["ANY", "NOUN", "VERB", "ADJ", "ADV", "PROPN"])
        
        if st.button("Search", type="primary"):
            try:
                if pos == "ANY":
                    tokens_res = supabase.table("tokens").select("*").eq("corpus_id", corpus_id).ilike("lemma", f"%{lemma}%").execute()
                else:
                    tokens_res = supabase.table("tokens").select("*").eq("corpus_id", corpus_id).eq("pos", pos).ilike("lemma", f"%{lemma}%").execute()
                
                if tokens_res.data:
                    tdf = pd.DataFrame(tokens_res.data)
                    st.info(f"Found {len(tdf)} matches")
                    st.dataframe(tdf[['word', 'lemma', 'pos', 'tag']].value_counts().reset_index(name='Count'), use_container_width=True)
            except Exception as e:
                st.error(f"Search error: {e}")

# === TAB 5: TRENDS & STATISTICS ===
with tabs[4]:
    st.markdown("### 📈 Corpus Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Text Length Distribution")
        try:
            trans_res = supabase.table("transcriptions").select("id, title").eq("corpus_id", corpus_id).execute()
            
            if trans_res.data:
                lengths = []
                for t in trans_res.data:
                    token_res = supabase.table("tokens").select("id", count="exact").eq("transcript_id", t['id']).execute()
                    count = token_res.count if hasattr(token_res, 'count') else len(token_res.data or [])
                    lengths.append({"text": t['title'], "tokens": count})
                
                if lengths:
                    len_df = pd.DataFrame(lengths).sort_values("tokens", ascending=True)
                    fig = px.barh(len_df, x="tokens", y="text", title="Tokens per Text")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    with col2:
        st.markdown("#### Corpus Diversity")
        try:
            tokens_res = supabase.table("tokens").select("lemma").eq("corpus_id", corpus_id).execute()
            
            if tokens_res.data:
                lemmas = [t['lemma'] for t in tokens_res.data]
                unique_lemmas = len(set(lemmas))
                total_tokens = len(lemmas)
                
                metrics = {
                    "Total Tokens": total_tokens,
                    "Unique Lemmas": unique_lemmas,
                    "Type/Token Ratio": unique_lemmas / total_tokens if total_tokens > 0 else 0,
                    "Richness Score": (unique_lemmas ** 2 / total_tokens) if total_tokens > 0 else 0
                }
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        st.metric(metric, f"{value:.4f}")
                    else:
                        st.metric(metric, f"{value:,}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("🔬 Linguistic Corpus Engine v2.0 | Powered by spaCy & Supabase")
