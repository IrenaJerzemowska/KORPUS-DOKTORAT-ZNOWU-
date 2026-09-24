import streamlit as st
import pandas as pd
import numpy as np
import re, os, io, json, math, unicodedata, html
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
st.set_page_config(page_title="Korpus językowy | Linguistic Corpus Engine", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")


# ============================================================
# INTERFACE LANGUAGE: Polish by default, with English toggle
# ============================================================
PL_TRANSLATIONS = {
    "Polish": "Polski", "English": "Angielski", "Interface language": "Język interfejsu",
    "Linguistic Corpus Engine": "Korpus językowy",
    "Corpus Management": "Zarządzanie korpusem", "Create new corpus": "Utwórz nowy korpus",
    "Name": "Nazwa", "Description": "Opis", "Main language": "Język główny",
    "Polish + English bilingual": "Polski i angielski (korpus dwujęzyczny)",
    "Create corpus": "Utwórz korpus", "Active corpus": "Aktywny korpus",
    "Delete this corpus": "Usuń ten korpus", "No corpora yet. Create one below.": "Nie ma jeszcze korpusów. Utwórz pierwszy poniżej.",
    "Create or select a corpus in the sidebar to start.": "Aby rozpocząć, utwórz korpus lub wybierz go z panelu bocznego.",
    "I understand this deletes all texts and tokens": "Rozumiem, że ta operacja usunie wszystkie teksty i tokeny",
    "Enter a corpus name.": "Wpisz nazwę korpusu.",
    "Upload & Manage": "Przesyłanie i zarządzanie", "KWIC + Timestamps": "KWIC i znaczniki czasu",
    "Dictionaries": "Słowniki", "Frequency & Trends": "Częstość i trendy",
    "Neologisms & Keyness": "Neologizmy i kluczowość", "Semantic Drift & Keywords over Time": "Zmiana semantyczna i słowa kluczowe w czasie",
    "Loanwords & Anglicisms": "Zapożyczenia i anglicyzmy", "Stance / Sentiment / Modality": "Stanowisko, sentyment i modalność",
    "Statistics & Metadata": "Statystyki i metadane", "Statistics & Metadata": "Statystyki i metadane",
    "Texts": "Teksty", "Tokens": "Tokeny", "Types (lemmas)": "Typy (lematy)", "TTR": "Wskaźnik typ-token (TTR)",
    "Add texts": "Dodawanie tekstów", "Method:": "Sposób dodawania:", "Paste text": "Wklej tekst",
    "Upload file (txt/srt/vtt)": "Prześlij pliki (DOCX/TXT/SRT/VTT)",
    "Texts in this corpus (edit / delete)": "Teksty w tym korpusie (edytuj / usuń)",
    "Title *": "Tytuł *", "Title": "Tytuł", "Video URL (YouTube link enables the timestamp viewer)": "Adres URL filmu (link YouTube włącza odtwarzacz ze znacznikami czasu)",
    "Video URL for these files (optional)": "Adres URL filmu dla tych plików (opcjonalnie)", "Video URL": "Adres URL filmu",
    "Transcript language (choose separately from the corpus default)": "Język transkrypcji (wybierz niezależnie od ustawienia korpusu)",
    "Transcript language": "Język transkrypcji", "Text content *": "Treść tekstu *", "Clean text": "Oczyszczony tekst",
    "Add to corpus": "Dodaj do korpusu", "Add files to corpus": "Dodaj pliki do korpusu",
    "Upload .docx / .txt / .srt / .vtt (batch: select multiple)": "Prześlij pliki .docx / .txt / .srt / .vtt (możesz wybrać wiele)",
    "Video ID": "Identyfikator filmu", "Video ID *": "Identyfikator filmu *", "Channel / Source": "Kanał / źródło", "Channel / Source *": "Kanał / źródło *",
    "Publish date": "Data publikacji", "Publish date *": "Data publikacji *", "Duration (seconds)": "Czas trwania (sekundy)", "Duration (seconds) *": "Czas trwania (sekundy) *",
    "Main speaker(s)": "Główny mówca / główni mówcy", "Genre": "Gatunek", "Language": "Język",
    "interview": "wywiad", "podcast": "podcast", "vlog": "wideoblog", "news": "wiadomości", "lecture": "wykład", "live stream": "transmisja na żywo", "other": "inne",
    "No texts yet.": "Nie dodano jeszcze tekstów.", "Add dictionary entry": "Dodaj wpis do słownika",
    "Dictionaries (lexicon of sentiment / modality / stance)": "Słowniki (leksykon sentymentu / modalności / stanowiska)",
    "These entries are stored in the database and shared with everyone who opens the app. Add your own terms.": "Wpisy są przechowywane w bazie i widoczne dla wszystkich użytkowników aplikacji. Możesz dodawać własne terminy.",
    "Term / phrase": "Termin / wyrażenie", "Term": "Termin", "Category": "Kategoria", "Value (for sentiment: -1..1)": "Wartość (dla sentymentu: od -1 do 1)",
    "Add entry": "Dodaj wpis", "Delete entries": "Usuń wpisy", "Confirm delete": "Potwierdź usunięcie", "Add": "Dodaj",
    "positive": "pozytywny", "negative": "negatywny", "epistemic": "epistemiczny", "deontic": "deontyczny", "stance_agree": "zgoda ze stanowiskiem", "stance_disagree": "brak zgody ze stanowiskiem", "custom": "własna kategoria",
    "word": "forma wyrazowa", "lemma": "lemat", "any": "dowolny", "agree-lean": "przewaga zgody", "disagree-lean": "przewaga niezgody",
    "Frequency & time-sliced trends": "Częstość i trendy w podziale na okresy", "Level:": "Poziom:", "Time slice granularity": "Jednostka podziału czasu",
    "Time slice for emergence": "Jednostka czasu dla analizy nowych słów", "Time granularity for anglicism trends": "Jednostka czasu dla trendów anglicyzmów",
    "Year": "Rok", "Quarter": "Kwartał", "Month": "Miesiąc", "Show top N": "Pokaż pierwszych N wyników",
    "Terms to track over time (empty = top 10 overall)": "Terminy śledzone w czasie (puste pole = 10 najczęstszych)",
    "Download full frequency list (CSV)": "Pobierz pełną listę częstości (CSV)",
    "KWIC concordance": "Konkordancja KWIC", "Match on:": "Szukaj według:", "Query (exact word, or wildcard with *):": "Zapytanie (dokładny wyraz lub symbol wieloznaczny *):",
    "Context words": "Liczba wyrazów kontekstu", "Max hits": "Maksymalna liczba trafień", "Run KWIC": "Uruchom KWIC", "Download KWIC as CSV": "Pobierz KWIC jako CSV",
    "Video timestamp viewer": "Odtwarzacz wideo ze znacznikami czasu", "Jump to hit:": "Przejdź do trafienia:",
    "Neologism / emergence spotter & keyness": "Wykrywanie neologizmów i nowych zjawisk oraz kluczowość", "Emergent words (late vs early period)": "Nowe wyrazy (okres późniejszy względem wcześniejszego)",
    "Keyness (log-likelihood, this corpus vs reference)": "Kluczowość (log-wiarygodność, korpus badany względem referencyjnego)",
    "Reference corpus (baseline)": "Korpus referencyjny (punkt odniesienia)", "Min frequency in target corpus": "Minimalna częstość w korpusie badanym", "Download keyness (CSV)": "Pobierz wyniki kluczowości (CSV)",
    "Semantic drift & keyword-in-time tracker": "Zmiana semantyczna i śledzenie słów kluczowych w czasie", "Granularity": "Jednostka czasu",
    "Keywords in time (distinctive terms per slice)": "Słowa kluczowe w czasie (charakterystyczne terminy dla okresu)",
    "Semantic drift: collocates of a word over time": "Zmiana semantyczna: kolokaty wyrazu w czasie", "Word to track collocates for:": "Wyraz, którego kolokaty chcesz śledzić:",
    "Collocate span (words each side)": "Zakres kolokacji (liczba wyrazów z każdej strony)", "Compute drift": "Oblicz zmianę semantyczną",
    "Loanword & anglicism tracker (Polish)": "Śledzenie zapożyczeń i anglicyzmów w polszczyźnie", "Domain (e.g. tech, social media)": "Dziedzina (np. technologia, media społecznościowe)",
    "Download anglicisms (CSV)": "Pobierz anglicyzmy (CSV)", "Stance, sentiment & modality filter": "Filtr stanowiska, sentymentu i modalności",
    "Filter transcripts by stance / sentiment / modality profile:": "Filtruj transkrypcje według profilu stanowiska, sentymentu lub modalności:",
    "Sentiment tilt": "Nastawienie sentymentu", "Min epistemic markers (hedging)": "Min. liczba wykładników epistemicznych (asekuracja)", "Min deontic markers (obligation)": "Min. liczba wykładników deontycznych (obowiązek)",
    "Stance": "Stanowisko", "any": "dowolne", "positive-lean": "przewaga pozytywnego", "negative-lean": "przewaga negatywnego", "agree-lean": "przewaga zgody", "disagree-lean": "przewaga niezgody",
    "Corpus statistics": "Statystyki korpusu", "Tokens per text": "Tokeny w tekście", "Vocabulary richness": "Bogactwo słownictwa", "Metadata coverage": "Kompletność metadanych", "POS distribution": "Rozkład części mowy",
    "Download metadata table (CSV)": "Pobierz tabelę metadanych (CSV)", "Guiraud index": "Indeks Guirauda", "Relative frequency (%)": "Częstość względna (%)",
    "Text": "Tekst", "Left": "Lewy kontekst", "Node": "Szukany wyraz", "Right": "Prawy kontekst", "POS": "Część mowy", "Timestamp": "Znacznik czasu",
    "Count": "Liczba", "Frequency": "Częstość", "Rel. freq (per 1M)": "Częstość względna (na milion)", "per_million": "Na milion", "slice": "Okres", "term": "Termin", "Co-occurrences": "Współwystąpienia", "Collocate": "Kolokat", "Keyword": "Słowo kluczowe", "Slice": "Okres", "Domain": "Dziedzina", "Count in corpus": "Liczba w korpusie", "Positive": "Pozytywne", "Negative": "Negatywne", "Epistemic": "Epistemiczne", "Deontic": "Deontyczne", "Agree": "Zgoda", "Disagree": "Niezgoda", "Mandatory": "Obowiązkowe", "Filled": "Uzupełnione", "Guiraud index": "Indeks Guirauda",
    "POS distribution": "Rozkład części mowy", "Lemma": "Lemat", "Word": "Wyraz", "Count in corpus": "Liczba w korpusie", "emergence": "Wzrost", "early": "Wcześniej", "late": "Później", "Target": "Badany", "Reference": "Referencyjny", "LL": "Wartość LL",
    "No matching texts yet.": "Nie znaleziono jeszcze pasujących tekstów.", "Upload texts first.": "Najpierw prześlij teksty.", "No terms reached LL >= 6.63 (p < 0.01). Lower the min frequency.": "Żaden termin nie osiągnął LL >= 6,63 (p < 0,01). Zmniejsz minimalną częstość.",
    "Add a YouTube URL to a transcript to embed the player here.": "Dodaj adres YouTube do transkrypcji, aby wyświetlić tutaj odtwarzacz.", "Counts come from the stored lexicon; edit it in the Dictionaries tab to refine the filter.": "Liczby pochodzą z zapisanego leksykonu; edytuj go w zakładce Słowniki, aby doprecyzować filtr.",
    "These entries are stored in the database and shared with everyone who opens the app. Add your own terms.": "Wpisy są przechowywane w bazie i widoczne dla wszystkich użytkowników aplikacji. Możesz dodawać własne terminy.",
    "Database error": "Błąd bazy danych", "Error saving transcript": "Błąd zapisu transkrypcji", "Failed to insert transcript (check that schema.sql was run).": "Nie udało się dodać transkrypcji (sprawdź, czy uruchomiono schema.sql).",
    "Title and text are required.": "Tytuł i tekst są wymagane.", "Missing mandatory metadata:": "Brak wymaganych metadanych:", "Tokenizing and tagging...": "Tokenizacja i oznaczanie...",
    "Add a YouTube URL to a transcript to embed the player here.": "Dodaj adres YouTube do transkrypcji, aby wyświetlić tutaj odtwarzacz.",
    "Delete": "Usuń", "Save changes": "Zapisz zmiany", "Re-tokenize": "Ponownie tokenizuj", "Text and tokens re-indexed.": "Tekst i tokeny zostały ponownie zaindeksowane.", "Saved.": "Zapisano.",
    "Text and tokens re-indexed.": "Tekst i tokeny zostały ponownie zaindeksowane.", "No texts yet.": "Nie dodano jeszcze tekstów.",
    "Type/Token Ratio": "Wskaźnik typów do tokenów", "Relative Freq (%)": "Częstość względna (%)", "Total Tokens": "Łączna liczba tokenów", "Unique Lemmas": "Unikalne lematy",
    "Powered by spaCy & Supabase": "Technologie: spaCy i Supabase", "Corpus Engine": "Silnik korpusowy",
    "Polish + English (Polish POS model)": "Polski i angielski (model anotacji polskiej)",
    "English + Polish (English POS model)": "Angielski i polski (model anotacji angielskiej)",
    "Upload file": "Prześlij plik", "Regex": "Wyrażenie regularne", "POS Tag": "Część mowy",
    "Lemma + POS": "Lemat i część mowy", "Search": "Szukaj", "Analyze": "Analizuj", "Download as CSV": "Pobierz jako CSV",
    "Found": "Znaleziono", "matching texts": "pasujących tekstów", "No matches found": "Nie znaleziono trafień",
    "No publish_date values set. Fill the mandatory 'Publish date' metadata to enable trends.": "Nie ustawiono dat publikacji. Uzupełnij wymagane metadane „Data publikacji”, aby włączyć analizę trendów.",
    "Needs publish_date metadata on at least two time slices.": "Do analizy potrzebne są daty publikacji dla co najmniej dwóch okresów.",
    "Needs publish_date metadata on at least two time slices.": "Do analizy potrzebne są daty publikacji dla co najmniej dwóch okresów.",
    "Empty": "Puste", "no date": "brak daty", "no channel": "brak kanału", "unknown": "nieznany", "general": "ogólne",
    "Corpus diversity": "Różnorodność korpusu", "POS": "Część mowy", "Frequency": "Częstość", "Count": "Liczebność",
    "Left": "Kontekst lewy", "Right": "Kontekst prawy", "Keyword": "Słowo kluczowe", "Text": "Tekst",
    "Score": "Wynik", "Relative frequency": "Częstość względna", "Type/Token Ratio": "Wskaźnik typów do tokenów",
    "TTR": "Wskaźnik typów do tokenów (TTR)", "Tokenizing and tagging...": "Tokenizacja i anotacja...",
    "Create a second corpus (e.g. your small test corpus or a general reference corpus) to enable neologism spotting and keyness.": "Utwórz drugi korpus, np. mały korpus testowy lub korpus referencyjny, aby analizować neologizmy i kluczowość.",
    "A stored list of English-origin items used in Polish. Extend it below; it persists for everyone.": "Lista angielskiego pochodzenia używanych w polszczyźnie. Możesz ją rozszerzyć; zmiany są zapisywane dla wszystkich.",
    "Add terms in the Dictionaries tab first.": "Najpierw dodaj terminy w zakładce Słowniki.",
    "Upload texts first.": "Najpierw prześlij teksty.", "No texts yet.": "Nie dodano jeszcze tekstów.",
    "Corpus Management": "Zarządzanie korpusem", "Create new corpus": "Utwórz nowy korpus",
    "Dictionary generated from uploaded texts": "Słownik generowany na podstawie przesłanych tekstów",
    "This corpus-derived word list updates from the texts stored in the corpus. It reports observed forms or lemmas, frequencies, and document coverage. It does not invent definitions; add definitions manually in your research notes or lexicon.": "Ta lista słów jest tworzona na podstawie tekstów zapisanych w korpusie. Zawiera zaobserwowane formy lub lematy, częstość oraz liczbę tekstów, w których występują. Definicje nie są generowane automatycznie; dodaj je ręcznie w notatkach badawczych lub leksykonie.",
    "Upload texts first to generate a corpus dictionary.": "Aby wygenerować słownik korpusowy, najpierw prześlij teksty.",
    "Dictionary language": "Język słownika", "All transcript languages": "Wszystkie języki transkrypcji", "Mixed Polish-English": "Polski i angielski (tekst mieszany)",
    "Entry type": "Typ hasła", "Minimum frequency": "Minimalna częstość", "Include common function words": "Uwzględnij częste wyrazy funkcyjne",
    "Filter entries": "Filtruj hasła", "Type a word or part of a word": "Wpisz wyraz lub jego fragment",
    "No entries match these filters. Lower the minimum frequency or include common words.": "Żadne hasło nie spełnia wybranych kryteriów. Zmniejsz minimalną częstość lub uwzględnij częste wyrazy.",
    "Maximum dictionary entries shown": "Maksymalna liczba wyświetlanych haseł", "Dictionary entries": "Liczba haseł słownika",
    "Download corpus dictionary (CSV)": "Pobierz słownik korpusowy (CSV)", "Show concordance examples for an entry": "Pokaż przykłady konkordancji dla hasła", "Show examples": "Pokaż przykłady",
    "No indexed contexts are available for this entry yet.": "Brak zaindeksowanych kontekstów dla tego hasła.",
    "Research lexicon: sentiment, modality, and stance": "Leksykon badawczy: sentyment, modalność i stanowisko",
    "The corpus-generated dictionary above is built from your uploaded texts. This editable lexicon stores manually classified terms used by the stance and sentiment filters.": "Powyższy słownik korpusowy powstaje na podstawie przesłanych tekstów. Ten edytowalny leksykon przechowuje ręcznie sklasyfikowane terminy używane w filtrach stanowiska i sentymentu.",
    "Entry": "Hasło", "Language": "Język", "Texts": "Teksty", "Per million": "Na milion", "Frequency": "Częstość", "Left context": "Kontekst lewy", "Right context": "Kontekst prawy",
    "No trend points can be plotted yet. Add dated transcripts containing searchable words, or change the selected terms.": "Brak punktów do wykreślenia trendu. Dodaj transkrypcje z datami i tekstem albo zmień wybrane terminy.",
    "Could not draw the trend chart for this selection: ": "Nie udało się narysować wykresu trendu dla tego wyboru: ",
}
PL_TRANSLATIONS.update({
    "Main speaker(s)": "Główny mówca / główni mówcy", "Genre": "Gatunek", "Language": "Język",
    "Linguistic Corpus Engine v3.0 | Streamlit + spaCy + Supabase. Data persists in the cloud; anyone with the app link can browse, search and analyze. Editing/deleting is available to all viewers; use Supabase RLS if you need to restrict this.": "Korpus językowy v3.0 | Streamlit + spaCy + Supabase. Dane są przechowywane w chmurze. Każda osoba z linkiem może przeglądać, wyszukiwać i analizować korpus.",
    "Include common function words": "Uwzględnij częste wyrazy funkcyjne", "Minimum frequency": "Minimalna częstość", "Filter entries": "Filtruj hasła", "Type a word or part of a word": "Wpisz wyraz lub jego fragment",
    "Dictionary language": "Język słownika", "Entry type": "Typ hasła", "Maximum dictionary entries shown": "Maksymalna liczba wyświetlanych haseł", "Dictionary entries": "Liczba haseł słownika",
    "Show concordance examples for an entry": "Pokaż przykłady konkordancji dla hasła", "Show examples": "Pokaż przykłady", "No indexed contexts are available for this entry yet.": "Brak zaindeksowanych kontekstów dla tego hasła.",
    "Research lexicon: sentiment, modality, and stance": "Leksykon badawczy: sentyment, modalność i stanowisko",
    "No entries match these filters. Lower the minimum frequency or include common words.": "Żadne hasło nie spełnia wybranych kryteriów. Zmniejsz minimalną częstość lub uwzględnij częste wyrazy.",
    "No token counts yet. Upload and index texts first.": "Brak zliczeń tokenów. Najpierw prześlij teksty i je zaindeksuj.",
    "No publish_date values set. Fill the mandatory 'Publish date' metadata to enable trends.": "Nie ustawiono dat publikacji. Uzupełnij wymagane pole „Data publikacji”, aby włączyć analizę trendów.",
    "Needs publish_date metadata on at least two time slices.": "Do analizy potrzebne są daty publikacji dla co najmniej dwóch okresów.",
    "Add terms in the Dictionaries tab first.": "Najpierw dodaj terminy w zakładce Słowniki.", "Upload texts first to generate a corpus dictionary.": "Aby utworzyć słownik korpusowy, najpierw prześlij teksty.",
    "None of the anglicisms from the list occurs in this corpus yet.": "Żaden z anglicyzmów z listy nie występuje jeszcze w tym korpusie.",
    "A stored list of English-origin items used in Polish. Extend it below; it persists for everyone.": "Lista wyrazów pochodzenia angielskiego używanych w polszczyźnie. Możesz ją rozszerzać; zmiany są zapisywane dla wszystkich.",
    "Counts come from the stored lexicon; edit it in the Dictionaries tab to refine the filter.": "Liczby pochodzą z zapisanego leksykonu; edytuj go w zakładce Słowniki, aby doprecyzować filtr.",
    "Compare collocates across slices: changing collocates = semantic drift (e.g. 'europa' drifting from 'unia' to 'kryzys').": "Porównaj kolokaty w kolejnych okresach. Zmieniające się kolokaty mogą wskazywać na zmianę semantyczną.",
    "Create a second corpus (e.g. your small test corpus or a general reference corpus) to enable neologism spotting and keyness.": "Utwórz drugi korpus, np. mały korpus testowy lub referencyjny, aby analizować neologizmy i kluczowość.",
    "Missing Supabase credentials. Add SUPABASE_URL and SUPABASE_KEY to your secrets (local: .streamlit/secrets.toml, cloud: Streamlit Cloud app settings).": "Brak danych logowania do Supabase. Ustaw adres projektu i klucz anon w sekretach lokalnych lub w ustawieniach Streamlit Cloud.",
    "Enter a corpus name.": "Wpisz nazwę korpusu.", "Title and text are required.": "Tytuł i tekst są wymagane.",
    "Failed to insert transcript (check that schema.sql was run).": "Nie udało się dodać transkrypcji. Sprawdź, czy uruchomiono skrypt schematu bazy.",
    "No trend points can be plotted yet. Add dated transcripts containing searchable words, or change the selected terms.": "Brak punktów do wykreślenia trendu. Dodaj datowane transkrypcje z tekstem albo zmień wybrane terminy.",
    "Add anglicism": "Dodaj anglicyzm", "Types": "Typy", "Set publish_date metadata to use time tracking.": "Uzupełnij datę publikacji, aby włączyć śledzenie zmian w czasie.",
    "Text Annotations": "Anotacja tekstu", "Annotate transcript": "Anotuj transkrypcję", "Selected transcript": "Wybrana transkrypcja",
    "Known anglicisms": "Znane anglicyzmy", "Annotate sentiment / modality / stance lexicon": "Anotuj leksykon sentymentu / modalności / stanowiska",
    "Lexicon categories to annotate": "Kategorie leksykonu do anotacji", "Annotate selected corpus-dictionary entries": "Anotuj wybrane hasła słownika korpusowego",
    "Corpus dictionary terms": "Hasła słownika korpusowego", "Annotate POS tags": "Anotuj części mowy", "POS tags to annotate": "Części mowy do anotacji",
    "Generate annotations": "Wygeneruj anotacje", "No text available for this transcript.": "Brak tekstu dla tej transkrypcji.",
    "No matches found for the selected annotation layers.": "Nie znaleziono trafień dla wybranych warstw anotacji.",
    "Download annotated Word document": "Pobierz anotowany dokument Word", "Download annotation spans (CSV)": "Pobierz zakresy anotacji (CSV)",
    "Possible anglicism candidates from Polish-English overlap": "Możliwe anglicyzmy na podstawie nakładania się korpusu polskiego i angielskiego",
    "These are candidates for manual review, not confirmed loanwords. Shared Polish/English forms can also be cognates or names.": "To kandydaci do ręcznej weryfikacji, nie potwierdzone zapożyczenia. Wspólne formy polsko-angielskie mogą być też kognatami lub nazwami własnymi.",
    "Minimum frequency in Polish texts": "Minimalna częstość w tekstach polskich", "Add selected candidates to anglicism list": "Dodaj wybranych kandydatów do listy anglicyzmów",
    "Possible candidates": "Możliwi kandydaci", "Polish frequency": "Częstość w polszczyźnie", "English frequency": "Częstość w angielszczyźnie", "Kandydat anglicyzmu": "Kandydat anglicyzmu",
    "Annotations are generated from the transcript, saved lexicon, known-anglicism list, selected corpus terms, and POS tags. The generated document is an export; source transcript stays unchanged.": "Anotacje powstają na podstawie transkrypcji, zapisanego leksykonu, listy anglicyzmów, wybranych haseł korpusowych i części mowy. Pobierany dokument jest kopią; oryginalna transkrypcja pozostaje bez zmian.",
    "Upload file (txt/srt/vtt)": "Prześlij plik (DOCX/TXT/SRT/VTT)", "Corpus statistics": "Statystyki korpusu",
    "The corpus-generated dictionary above is built from your uploaded texts. This editable lexicon stores manually classified terms used by the stance and sentiment filters.": "Powyższy słownik korpusowy powstaje na podstawie przesłanych tekstów. Ten edytowalny leksykon przechowuje ręcznie sklasyfikowane terminy używane w filtrach stanowiska i sentymentu.",
})

_TRANSLATION_PATTERNS = [
    (re.compile(r"^No collocates found for '(.+)' in this corpus\. Check the spelling \(the search is case-sensitive and matches the stored lowercase form\)\.$"), r"Nie znaleziono kolokatów dla „\1” w korpusie. Sprawdź pisownię; wyszukiwanie uwzględnia zapis z bazy."),
    (re.compile(r"^Added (\d+) candidate\(s\)\. Review them in the anglicism list before treating them as established loans\.$"), r"Dodano kandydatów: \1. Sprawdź ich na liście anglicyzmów przed uznaniem za zapożyczenia."),
    (re.compile(r"^spaCy could not initialize \((.*)\)\. Using regex tokenization without POS tags for this run\. Check the pinned compatible dependencies in requirements\.txt\.$"), r"Nie udało się uruchomić spaCy (\1). Tymczasowo używana jest tokenizacja bez oznaczania części mowy. Sprawdź zgodność pakietów w pliku requirements.txt."),
    (re.compile(r"^spaCy model '(.+)' could not load \((.*)\)\. Using regex tokenization without POS tags for this run\. Check the pinned compatible dependencies in requirements\.txt\.$"), r"Nie udało się wczytać modelu spaCy „\1” (\2). Tymczasowo używana jest tokenizacja bez oznaczania części mowy. Sprawdź zgodność pakietów w pliku requirements.txt."),
    (re.compile(r"^Could not draw the trend chart for this selection: (.*)$"), r"Nie udało się narysować wykresu trendu dla tego wyboru: \1"),
    (re.compile(r"^No data points to plot yet\. Check that the chosen terms occur in the selected period\.$"), "Brak punktów do wykreślenia. Sprawdź, czy wybrane terminy występują w tym okresie."),
    (re.compile(r"^Most frequent words$"), "Najczęstsze wyrazy"),
    (re.compile(r"^Most frequent lemmas$"), "Najczęstsze lematy"),
    (re.compile(r"^Frequency per million words \((.*?)ly\)$"), r"Częstość na milion wyrazów (\1)"),
    (re.compile(r"^Collocates of '(.+)' across time slices$"), r"Kolokaty wyrazu „\1” w kolejnych okresach"),
    (re.compile(r"^Anglicisms by frequency$"), "Anglicyzmy według częstości"),
    (re.compile(r"^Top anglicisms over time$"), "Najczęstsze anglicyzmy w czasie"),
    (re.compile(r"^Tokens per text$"), "Tokeny w poszczególnych tekstach"),
    (re.compile(r"^POS distribution$"), "Rozkład części mowy"),
    (re.compile(r"^#### (\d+) anglicisms found in this corpus$"), r"#### Liczba anglicyzmów w korpusie: \1"),
    (re.compile(r"^Mandatory fields still empty: (.*)$"), r"Nieuzupełnione wymagane pola: \1"),
    (re.compile(r"^(\d+) matching texts$"), r"Pasujące teksty: \1"),
    (re.compile(r"^Context: \.\.\. (.*)$"), r"Kontekst: ... \1"),
    (re.compile(r"^Corpus '(.+)' created\.$"), r"Utworzono korpus „\1”."),
    (re.compile(r"^Added (\d+) file\(s\)\.$"), r"Dodano plików: \1."),
    (re.compile(r"^Saved\. ([\d,]+) tokens indexed\.$"), r"Zapisano. Zaindeksowano tokenów: \1."),
    (re.compile(r"^Found (\d+) matching texts$"), r"Znaleziono pasujących tekstów: \1"),
    (re.compile(r"^Found (\d+) matches$"), r"Znaleziono trafień: \1"),
    (re.compile(r"^Found (\d+) anglicisms in this corpus$"), r"Liczba anglicyzmów w korpusie: \1"),
    (re.compile(r"^Found (\d+) tokens with POS=(.+)$"), r"Liczba tokenów z częścią mowy \2: \1"),
    (re.compile(r"^Error saving transcript: (.*)$"), r"Błąd zapisu transkrypcji: \1"),
    (re.compile(r"^Database error \[(.*?)\]: (.*)$"), r"Błąd bazy danych [\1]: \2"),
    (re.compile(r"^Default (.+) data could not be seeded (.*)$"), r"Nie udało się dodać danych domyślnych (\1): \2"),
]

def _translate_text(value, allow_fragments=False):
    if not isinstance(value, str):
        return value
    current_language = globals().get("CURRENT_INTERFACE_LANGUAGE", st.session_state.get("interface_language", "Polish"))
    if current_language not in ("Polish", "Polski"):
        return value
    for pattern, replacement in _TRANSLATION_PATTERNS:
        if pattern.match(value):
            return pattern.sub(replacement, value)
    # Translate exact UI strings and embedded headings, longest phrases first.
    if value in PL_TRANSLATIONS:
        return PL_TRANSLATIONS[value]
    if allow_fragments:
        for en in sorted(PL_TRANSLATIONS, key=len, reverse=True):
            if len(en) > 4 and en in value:
                value = value.replace(en, PL_TRANSLATIONS[en])
    return value

def _install_polish_interface():
    """Localize visible Streamlit labels without changing stored data or widget values."""
    try:
        from streamlit.delta_generator import DeltaGenerator
    except Exception:
        return
    methods = ["markdown", "title", "header", "subheader", "caption", "info", "warning", "error", "success", "exception",
               "text_input", "text_area", "number_input", "date_input", "selectbox", "multiselect", "radio", "checkbox", "button",
               "file_uploader", "metric", "tabs", "dataframe", "download_button", "spinner", "expander"]
    for method_name in methods:
        original = getattr(DeltaGenerator, method_name, None)
        if original is None or getattr(original, "_polish_i18n_wrapped", False):
            continue
        def make_wrapper(original_method, name):
            def wrapped(self, *args, **kwargs):
                args = list(args)
                if name == "tabs" and args:
                    args[0] = [_translate_text(x) for x in args[0]]
                elif name == "dataframe":
                    try:
                        import pandas as _pd
                        if args and isinstance(args[0], _pd.DataFrame):
                            args[0] = args[0].rename(columns=lambda x: _translate_text(str(x)))
                        elif isinstance(kwargs.get("data"), _pd.DataFrame):
                            kwargs["data"] = kwargs["data"].rename(columns=lambda x: _translate_text(str(x)))
                    except Exception:
                        pass
                else:
                    # Translate human-facing first argument (widget labels, alerts, headings).
                    if args and isinstance(args[0], str):
                        args[0] = _translate_text(args[0], allow_fragments=(name == "markdown"))
                    # Streamlit select/radio options retain their original values; only their display labels translate.
                    if name in ("selectbox", "multiselect", "radio"):
                        fmt = kwargs.get("format_func")
                        if fmt is None:
                            kwargs["format_func"] = lambda value: _translate_text(str(value))
                        else:
                            kwargs["format_func"] = lambda value, _fmt=fmt: _translate_text(str(_fmt(value)))
                    for key in ("help", "placeholder", "label_visibility"):
                        if isinstance(kwargs.get(key), str):
                            kwargs[key] = _translate_text(kwargs[key])
                return original_method(self, *args, **kwargs)
            wrapped._polish_i18n_wrapped = True
            return wrapped
        setattr(DeltaGenerator, method_name, make_wrapper(original, method_name))

_install_polish_interface()
# Default to Polish, but make an English option available for presentations.
CURRENT_INTERFACE_LANGUAGE = st.sidebar.selectbox(_translate_text("Interface language"), ["Polish", "English"], index=0, key="interface_language")


def safe_plotly_chart(factory_or_figure, **kwargs):
    """Create and display a Plotly chart without patching Streamlit internals."""
    try:
        figure = factory_or_figure() if callable(factory_or_figure) else factory_or_figure
        if globals().get("CURRENT_INTERFACE_LANGUAGE", st.session_state.get("interface_language", "Polish")) in ("Polish", "Polski"):
            try:
                if getattr(figure.layout, "title", None) and figure.layout.title.text:
                    figure.layout.title.text = _translate_text(figure.layout.title.text)
                for axis_name in ("xaxis", "yaxis", "legend"):
                    axis = getattr(figure.layout, axis_name, None)
                    if axis is not None and getattr(axis, "title", None) and getattr(axis.title, "text", None):
                        axis.title.text = _translate_text(axis.title.text)
            except Exception:
                pass
        st.plotly_chart(figure, **kwargs)
    except Exception as exc:
        msg_pl = f"Nie udało się wyświetlić wykresu ({type(exc).__name__}): {exc}"
        msg_en = f"This chart could not be displayed ({type(exc).__name__}): {exc}"
        st.warning(msg_pl if globals().get("CURRENT_INTERFACE_LANGUAGE", st.session_state.get("interface_language", "Polish")) in ("Polish", "Polski") else msg_en)

st.markdown(_translate_text("""
<style>
    .main-header { font-size: 2.2em; font-weight: bold; color: #1f77b4; margin-bottom: 0.3em; }
    .subheader { font-size: 1.15em; font-weight: bold; color: #333; margin-top: 0.8em; margin-bottom: 0.4em; }
    .kwic-node { font-weight: bold; background-color: #fff3cd; padding: 0.1em 0.35em; border-radius: 0.25em; }
    .ts-chip { font-size: 0.85em; color: #555; font-family: monospace; }
</style>
""", allow_fragments=True), unsafe_allow_html=True)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error(_translate_text("Missing Supabase credentials. Add SUPABASE_URL and SUPABASE_KEY to your secrets (local: .streamlit/secrets.toml, cloud: Streamlit Cloud app settings)."))
        st.stop()
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(_translate_text(f"Could not connect to Supabase: {e}"))
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
    "23514": "A database CHECK constraint rejected the transcript language. Run bilingual_language_fix.sql in Supabase to allow pl, en, pl-en and en-pl language labels.",
    "42P01": "The table does not exist. Run schema.sql in the Supabase SQL Editor first.",
    "PGRST301": "The API key was rejected. Check SUPABASE_KEY: it must be the anon public key from Supabase > Project Settings > API.",
}

def show_db_error(e):
    code = getattr(e, "code", "") or ""
    msg = getattr(e, "message", None) or str(e)
    st.error(_translate_text(f"Database error [{code or 'unknown'}]: {msg}"))
    for k, hint in HINTS.items():
        if code == k or (not code and k in msg):
            st.info(_translate_text(hint))
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
            st.error(_translate_text(f"Database error: {e}"))
            return []
    return inserted

def db_delete(table, column, value):
    try:
        supabase.table(table).delete().eq(column, value).execute()
    except APIError as e:
        show_db_error(e)
    except Exception as e:
        st.error(_translate_text(f"Database error: {e}"))

# ============================================================
# NLP (fix #6: spaCy models come from requirements.txt; graceful fallback)
# ============================================================
SPACY_MODELS = {"pl": "pl_core_news_sm", "en": "en_core_web_sm", "de": "de_core_web_sm", "fr": "fr_core_web_sm", "es": "es_core_web_sm", "zh": "zh_core_web_sm"}

TEXT_LANGUAGE_OPTIONS = {
    "Polish": "pl",
    "English": "en",
    "Polish + English (Polish POS model)": "pl-en",
    "English + Polish (English POS model)": "en-pl",
}

def primary_nlp_language(language: str) -> str:
    """Choose the dominant-language spaCy model for bilingual transcripts."""
    return (language or "pl").split("-")[0]


@st.cache_resource
def load_spacy_model(lang_code: str):
    model_name = SPACY_MODELS.get(lang_code, "en_core_web_sm")
    try:
        import spacy
        try:
            return spacy.load(model_name), model_name
        except Exception as e:
            st.warning(_translate_text(f"spaCy model '{model_name}' could not load ({e}). Using regex tokenization without POS tags for this run. Check the pinned compatible dependencies in requirements.txt."))
            return None, model_name
    except Exception as e:
        st.warning(_translate_text(f"spaCy could not initialize ({e}). Using regex tokenization without POS tags for this run. Check the pinned compatible dependencies in requirements.txt."))
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

def save_transcript(corpus_id: int, title: str, lang: str, raw_text: str, metadata: dict | None = None, video_url: str = "", nlp_lang: str | None = None):
    t_id = None
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
            st.error(_translate_text("Failed to insert transcript (check that schema.sql was run)."))
            return 0
        t_id = res[0]["id"]

        token_rows = []
        idx_base = 0
        all_tokens = []
        with st.spinner(_translate_text("Tokenizing and tagging...")):
            for seg_text, ts in segments:
                seg_clean = clean_and_normalize(seg_text)
                toks = tokenize(seg_clean, nlp_lang or primary_nlp_language(lang))
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
        # Avoid leaving a transcript row with no tokens if processing fails midway.
        if t_id is not None:
            try:
                supabase.table("tokens").delete().eq("transcript_id", t_id).execute()
                supabase.table("transcriptions").delete().eq("id", t_id).execute()
                clear_caches()
            except Exception:
                pass
        st.error(_translate_text(f"Error saving transcript: {e}"))
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
    ("video_url", "Video URL", "text", False),
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
            st.warning(_translate_text(f"Default {table} data could not be seeded ({code or 'database error'}): {getattr(e, 'message', str(e))}"))
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
        st.warning(_translate_text(f"Default data seeding could not complete ({code or 'database error'}): {getattr(e, 'message', str(e))}. Run schema_migrate.sql in Supabase."))

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


ANNOTATION_COLORS = {
    "Anglicyzm": "#ffe066", "Kandydat anglicyzmu": "#ffd8a8", "Sentyment pozytywny": "#8ce99a", "Sentyment negatywny": "#ffa8a8",
    "Modalność epistemiczna": "#99e9f2", "Modalność deontyczna": "#b197fc", "Stanowisko: zgoda": "#a9e34b",
    "Stanowisko: niezgoda": "#ff8787", "Hasło korpusowe": "#ffd8a8", "Część mowy": "#bac8ff",
}

def make_bar_figure(frame, x, y, title, orientation="v", color=None):
    """Build bars with graph_objects so chart availability doesn't depend on px.bar."""
    fig = go.Figure()
    if frame is None or frame.empty:
        return fig
    groups = [(None, frame)] if not color or color not in frame.columns else list(frame.groupby(color, dropna=False, sort=False))
    for name, group in groups:
        kwargs = {"orientation": "h"} if orientation == "h" else {}
        if name is not None:
            kwargs["name"] = str(name)
        fig.add_trace(go.Bar(x=group[x], y=group[y], **kwargs))
    fig.update_layout(title=title, barmode="group" if color else "relative", xaxis_title=x, yaxis_title=y)
    return fig

def find_annotation_spans(text, term_labels):
    """Locate exact words/phrases, allowing flexible whitespace and case."""
    spans = []
    seen = set()
    for term, label in sorted(term_labels, key=lambda item: len(item[0]), reverse=True):
        term = str(term).strip()
        if not term:
            continue
        body = re.escape(term).replace(r"\ ", r"\s+")
        pattern = re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE | re.UNICODE)
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), label, match.group(0))
            if key not in seen:
                spans.append({"start": match.start(), "end": match.end(), "term": match.group(0), "label": label})
                seen.add(key)
    return sorted(spans, key=lambda x: (x["start"], x["end"], x["label"]))

def annotation_html(text, spans):
    """Render escaped source text with overlapping annotation labels in colored highlights."""
    boundaries = sorted({0, len(text)} | {p for row in spans for p in (row["start"], row["end"])})
    parts = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = sorted({row["label"] for row in spans if row["start"] <= start and row["end"] >= end})
        chunk = html.escape(text[start:end]).replace("\n", "<br>")
        if active:
            color = ANNOTATION_COLORS.get(active[0], "#fff3bf")
            tip = html.escape(", ".join(active), quote=True)
            parts.append(f'<mark style="background-color:{color}; padding:1px 3px" title="{tip}">{chunk}</mark>')
        else:
            parts.append(chunk)
    return "".join(parts)

def annotation_docx_bytes(text, spans, title):
    """Create a color-highlighted Word copy with a compact annotation legend."""
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX
    color_enum = {
        "Anglicyzm": WD_COLOR_INDEX.YELLOW, "Kandydat anglicyzmu": WD_COLOR_INDEX.GRAY_25, "Sentyment pozytywny": WD_COLOR_INDEX.BRIGHT_GREEN,
        "Sentyment negatywny": WD_COLOR_INDEX.PINK, "Modalność epistemiczna": WD_COLOR_INDEX.TURQUOISE,
        "Modalność deontyczna": WD_COLOR_INDEX.VIOLET, "Stanowisko: zgoda": WD_COLOR_INDEX.BRIGHT_GREEN,
        "Stanowisko: niezgoda": WD_COLOR_INDEX.RED, "Hasło korpusowe": WD_COLOR_INDEX.YELLOW,
        "Część mowy": WD_COLOR_INDEX.BLUE,
    }
    doc = Document()
    doc.add_heading(f"Anotacja korpusowa: {title}", level=1)
    labs = sorted({r["label"] for r in spans})
    if labs:
        doc.add_paragraph("Legenda: " + "; ".join(labs))
    boundaries = sorted({0, len(text)} | {p for row in spans for p in (row["start"], row["end"])})
    # Preserve paragraph breaks while retaining all matched offsets.
    line_start = 0
    for line in text.splitlines(keepends=True) or [text]:
        content = line.rstrip("\r\n")
        para = doc.add_paragraph()
        line_end = line_start + len(content)
        points = sorted({line_start, line_end} | {p for p in boundaries if line_start < p < line_end})
        for a, b in zip(points, points[1:]):
            segment = text[a:b]
            active = sorted({r["label"] for r in spans if r["start"] <= a and r["end"] >= b})
            run = para.add_run(segment)
            if active:
                run.font.highlight_color = color_enum.get(active[0], WD_COLOR_INDEX.YELLOW)
        if line.endswith("\n"):
            para.add_run("\n")
        line_start += len(line)
    if not text:
        doc.add_paragraph("")
    out = io.BytesIO()
    doc.save(out)
    out.seek(0)
    return out.getvalue()
# ============================================================
# SIDEBAR: corpus management
# ============================================================
with st.sidebar:
    st.markdown(_translate_text("## Corpus Management", allow_fragments=True))
    st.session_state.setdefault("corpus_id", None)

    corpora = cached_corpora()
    if not corpora:
        st.info(_translate_text("No corpora yet. Create one below."))
        st.session_state.corpus_id = None
    else:
        labels = {c["id"]: c["name"] for c in corpora}
        current = st.session_state.corpus_id if st.session_state.corpus_id in labels else list(labels)[0]
        sel = st.selectbox(_translate_text("Active corpus"), options=list(labels), format_func=lambda i: labels[i], index=list(labels).index(current))
        st.session_state.corpus_id = sel
        if st.button(_translate_text("Delete this corpus"), type="secondary", use_container_width=True):
            st.session_state.confirm_delete = True
        if st.session_state.get("confirm_delete"):
            if st.checkbox(_translate_text("I understand this deletes all texts and tokens")):
                if st.button(_translate_text("Confirm delete"), type="primary"):
                    delete_corpus(sel)
                    st.session_state.corpus_id = None
                    st.session_state.confirm_delete = False
                    st.rerun()

    st.divider()
    st.markdown(_translate_text("### Create new corpus", allow_fragments=True))
    new_name = st.text_input(_translate_text("Name"), key="new_corpus_name", placeholder="e.g. Polish YouTube Podcasts")
    new_desc = st.text_area(_translate_text("Description"), key="new_corpus_desc", height=70)
    new_lang = st.selectbox(_translate_text("Main language"), ["pl", "en", "pl-en"], format_func=lambda l: {"pl": "Polish", "en": "English", "pl-en": "Polish + English bilingual"}[l])
    if st.button(_translate_text("Create corpus"), type="primary", use_container_width=True):
        if not new_name.strip():
            st.error(_translate_text("Enter a corpus name."))
        elif any(c["name"].lower() == new_name.strip().lower() for c in corpora):
            st.error(_translate_text(f"A corpus named '{new_name.strip()}' already exists. Pick another name."))
        else:
            created = db_insert("corpora", {"name": new_name.strip(), "description": new_desc, "language": new_lang, "created_at": datetime.now().isoformat(), "token_count": 0})
            if created:
                clear_caches()
                st.success(_translate_text(f"Corpus '{new_name.strip()}' created."))
                st.rerun()

corpus_id = st.session_state.get("corpus_id")
if not corpus_id:
    st.markdown(_translate_text("<h1 class='main-header'>Linguistic Corpus Engine</h1>", allow_fragments=True), unsafe_allow_html=True)
    st.warning(_translate_text("Create or select a corpus in the sidebar to start."))
    st.stop()

ensure_seed_data()
transcripts = cached_transcripts(corpus_id)
CORPUS_LANG = next((c.get("language", "pl") for c in cached_corpora() if c["id"] == corpus_id), "pl")

# header metrics
total_tokens = corpus_size(corpus_id)
c1, c2, c3, c4 = st.columns(4)
c1.metric(_translate_text("Texts"), len(transcripts))
c2.metric(_translate_text("Tokens"), f"{total_tokens:,}")
types = len(corpus_word_freq(corpus_id, "lemma"))
c3.metric(_translate_text("Types (lemmas)"), f"{types:,}")
c4.metric(_translate_text("TTR"), f"{types / total_tokens:.3f}" if total_tokens else "n/a")
st.divider()

# ============================================================
# TAB 1: UPLOAD & MANAGE (fix #3: edit + delete + metadata schema)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    _translate_text(label) for label in [
        "Upload & Manage", "KWIC + Timestamps", "Dictionaries", "Frequency & Trends",
        "Neologisms & Keyness", "Semantic Drift & Keywords over Time", "Loanwords & Anglicisms",
        "Stance / Sentiment / Modality", "Statistics & Metadata", "Text Annotations",
    ]
])

with tab1:
    st.markdown(_translate_text("<h3 class='subheader'>Add texts</h3>", allow_fragments=True), unsafe_allow_html=True)
    meta_fields = cached_metadata_fields()
    upload_method = st.radio(_translate_text("Method:"), ["Paste text", "Upload file (txt/srt/vtt)"], horizontal=True)

    def choose_transcript_language(key):
        default_lang = CORPUS_LANG if CORPUS_LANG in TEXT_LANGUAGE_OPTIONS.values() else "pl"
        labels = list(TEXT_LANGUAGE_OPTIONS.keys())
        values = list(TEXT_LANGUAGE_OPTIONS.values())
        default_index = values.index(default_lang) if default_lang in values else 0
        selected = st.selectbox(_translate_text("Transcript language (choose separately from the corpus default)"), labels, index=default_index, key=key)
        return TEXT_LANGUAGE_OPTIONS[selected]

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
                    v = st.selectbox(_translate_text(f["label"] + (" *" if f["mandatory"] else "")), opts + ["(none)"], key=widget_key)
                    if v not in (None, "(none)"):
                        md[f["field_name"]] = v
                elif f["field_type"] == "date":
                    v = st.date_input(_translate_text(f["label"] + (" *" if f["mandatory"] else "")), value=None, key=widget_key)
                    if v:
                        md[f["field_name"]] = v.isoformat()
                elif f["field_type"] == "number":
                    v = st.number_input(_translate_text(f["label"] + (" *" if f["mandatory"] else "")), min_value=0, step=1, key=widget_key)
                    if v is not None:
                        md[f["field_name"]] = int(v)
                else:
                    v = st.text_input(_translate_text(f["label"] + (" *" if f["mandatory"] else "")), key=widget_key)
                    if v.strip():
                        md[f["field_name"]] = v.strip()
            fidx += 1
        return md

    if upload_method == "Paste text":
        title = st.text_input(_translate_text("Title *"), key="paste_title")
        transcript_lang = choose_transcript_language("paste_language")
        video_url = st.text_input(_translate_text("Video URL (YouTube link enables the timestamp viewer)"), key="paste_url")
        md = metadata_form("paste")
        missing = [f["label"] for f in meta_fields if f["mandatory"] and f["field_name"] not in md and f["field_name"] not in ("language", "video_url")]
        text_input = st.text_area(_translate_text("Text content *"), height=220, key="paste_text", placeholder="Paste transcript. Inline [mm:ss] markers or SRT/VTT cues are detected automatically for timestamps.")
        if st.button(_translate_text("Add to corpus"), type="primary", use_container_width=True):
            if not title.strip() or not text_input.strip():
                st.error(_translate_text("Title and text are required."))
            elif missing:
                st.error(_translate_text("Missing mandatory metadata: " + ", ".join(missing)))
            else:
                n = save_transcript(corpus_id, title.strip(), transcript_lang, text_input, metadata=md, video_url=video_url, nlp_lang=primary_nlp_language(transcript_lang))
                if n:
                    st.success(_translate_text(f"Saved. {n:,} tokens indexed."))
                    st.rerun()
    else:
        up = st.file_uploader(_translate_text("Upload .docx / .txt / .srt / .vtt (batch: select multiple)"), type=["docx", "txt", "srt", "vtt"], accept_multiple_files=True)
        if up:
            transcript_lang = choose_transcript_language("file_language")
            md = metadata_form("file")
            missing = [f["label"] for f in meta_fields if f["mandatory"] and f["field_name"] not in md and f["field_name"] not in ("language", "video_url")]
            if missing:
                st.warning(_translate_text("Mandatory fields still empty: " + ", ".join(missing)))
            video_url = st.text_input(_translate_text("Video URL for these files (optional)"), key="file_url")
            if st.button(_translate_text("Add files to corpus"), type="primary", use_container_width=True):
                added = 0
                for f in up:
                    try:
                        if f.name.lower().endswith(".docx"):
                            from docx import Document
                            from docx.oxml.text.paragraph import CT_P
                            from docx.oxml.table import CT_Tbl
                            from docx.text.paragraph import Paragraph
                            from docx.table import Table
                            doc = Document(io.BytesIO(f.getvalue()))
                            blocks = []
                            # Walk the document body in order so paragraphs and tables stay in sequence.
                            for element in doc.element.body.iterchildren():
                                if isinstance(element, CT_P):
                                    text = Paragraph(element, doc).text.strip()
                                    if text:
                                        blocks.append(text)
                                elif isinstance(element, CT_Tbl):
                                    table = Table(element, doc)
                                    for row in table.rows:
                                        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                                        row_text = " | ".join(cell for cell in cells if cell)
                                        if row_text:
                                            blocks.append(row_text)
                            content = "\n".join(blocks)
                            if not content.strip():
                                st.warning(_translate_text(f"{f.name}: no extractable text found, skipped."))
                                continue
                        else:
                            content = f.getvalue().decode("utf-8-sig", errors="replace")
                    except Exception as e:
                        st.error(_translate_text(f"{f.name}: could not extract text ({e})"))
                        continue
                    tname = re.sub(r"\.(docx|txt|srt|vtt)$", "", f.name, flags=re.I)
                    n = save_transcript(corpus_id, tname, transcript_lang, content, metadata=md, video_url=video_url, nlp_lang=primary_nlp_language(transcript_lang))
                    added += 1 if n else 0
                if added:
                    st.success(_translate_text(f"Added {added} file(s)."))
                    st.rerun()

    st.divider()
    st.markdown(_translate_text("<h3 class='subheader'>Texts in this corpus (edit / delete)</h3>", allow_fragments=True), unsafe_allow_html=True)
    if transcripts:
        for t in transcripts:
            with st.expander(_translate_text(f"{t['title']}  ({(t.get('publish_date') or 'no date')} | {(t.get('channel') or 'no channel')})")):
                e_title = st.text_input(_translate_text("Title"), value=t["title"], key=f"et_{t['id']}")
                current_lang = t.get("language") or CORPUS_LANG
                lang_values = list(TEXT_LANGUAGE_OPTIONS.values())
                lang_labels = list(TEXT_LANGUAGE_OPTIONS.keys())
                selected_lang = current_lang if current_lang in lang_values else "pl"
                e_lang_label = st.selectbox(_translate_text("Transcript language"), lang_labels, index=lang_values.index(selected_lang), key=f"elang_{t['id']}")
                e_lang = TEXT_LANGUAGE_OPTIONS[e_lang_label]
                e_url = st.text_input(_translate_text("Video URL"), value=t.get("video_url") or "", key=f"eu_{t['id']}")
                e_text = st.text_area(_translate_text("Clean text"), value=t.get("clean_text") or "", height=140, key=f"ex_{t['id']}")
                e_md = {}
                for f in meta_fields:
                    cur = t.get(f["field_name"])
                    if f["field_type"] == "select":
                        opts = (f.get("options") or []) + ["(none)"]
                        v = st.selectbox(_translate_text(f["label"]), opts, index=opts.index(cur) if cur in opts else len(opts) - 1, key=f"em_{t['id']}_{f['field_name']}")
                        e_md[f["field_name"]] = None if v == "(none)" else v
                    elif f["field_type"] == "date":
                        v = st.date_input(_translate_text(f["label"]), value=pd.to_datetime(cur).date() if cur else None, key=f"em_{t['id']}_{f['field_name']}")
                        e_md[f["field_name"]] = v.isoformat() if v else None
                    else:
                        e_md[f["field_name"]] = st.text_input(_translate_text(f["label"]), value=str(cur) if cur else "", key=f"em_{t['id']}_{f['field_name']}")
                cA, cB, cC = st.columns(3)
                if cA.button(_translate_text("Save changes"), key=f"sv_{t['id']}"):
                    updates = {"title": e_title, "video_url": e_url or None, "language": e_lang}
                    updates.update({k: v for k, v in e_md.items()})
                    retext = e_text != (t.get("clean_text") or "")
                    if retext:
                        updates["clean_text"] = e_text
                        updates["raw_text"] = e_text
                    update_transcript(t["id"], updates)
                    if retext:
                        n = save_transcript(corpus_id, e_title, e_lang, e_text, metadata=updates, video_url=e_url, nlp_lang=primary_nlp_language(e_lang))
                        delete_transcript(t["id"])  # old tokens
                        st.success(_translate_text("Text and tokens re-indexed."))
                    else:
                        st.success(_translate_text("Saved."))
                    st.rerun()
                if cB.button(_translate_text("Re-tokenize"), key=f"rt_{t['id']}"):
                    delete_transcript(t["id"])
                    save_transcript(corpus_id, t["title"], t.get("language") or CORPUS_LANG, t.get("raw_text") or t.get("clean_text") or "", metadata={k: t.get(k) for k in [f["field_name"] for f in meta_fields]}, video_url=t.get("video_url") or "", nlp_lang=primary_nlp_language(t.get("language") or CORPUS_LANG))
                    st.rerun()
                if cC.button(_translate_text("Delete"), key=f"dl_{t['id']}", type="primary"):
                    delete_transcript(t["id"])
                    st.rerun()
    else:
        st.info(_translate_text("No texts yet."))

# ============================================================
# TAB 2: KWIC + TIMESTAMP VIEWER (fix #2: single query, fix #4: ts)
# ============================================================
with tab2:
    st.markdown(_translate_text("<h3 class='subheader'>KWIC concordance</h3>", allow_fragments=True), unsafe_allow_html=True)
    kw_all = corpus_word_freq(corpus_id, "word")
    lm_all = corpus_word_freq(corpus_id, "lemma")
    if not kw_all:
        st.info(_translate_text("Upload texts first."))
    else:
        cL, cR = st.columns([3, 1])
        with cL:
            mode = st.radio(_translate_text("Match on:"), ["word", "lemma"], horizontal=True, key="kwic_mode")
            query = st.text_input(_translate_text("Query (exact word, or wildcard with *):"), placeholder="e.g. polityk* or 'jest'")
        with cR:
            window = st.slider("Context words", 2, 15, 5, key="kwic_win")
            max_hits = st.slider("Max hits", 10, 500, 100, key="kwic_max")
        pattern = "^" + re.escape(query.lower()).replace(r"\*", ".*") + "$" if query and "*" in query else (query.lower() if query else None)

        if st.button(_translate_text("Run KWIC"), type="primary") and pattern:
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
            st.info(_translate_text(f"{len(hits)} matching texts"))
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
                st.download_button(_translate_text("Download KWIC as CSV"), kwic_df.drop(columns=["token_index", "transcript_id"]).to_csv(index=False), file_name="kwic.csv", mime="text/csv")

                # timestamp viewer
                st.divider()
                st.markdown(_translate_text("<h3 class='subheader'>Video timestamp viewer</h3>", allow_fragments=True), unsafe_allow_html=True)
                url_for_ts = None
                for t in transcripts:
                    if t["id"] == kwic_df.iloc[0]["transcript_id"]:
                        url_for_ts = t.get("video_url")
                if url_for_ts:
                    yt = youtube_id(url_for_ts)
                else:
                    yt = None
                if not yt:
                    st.caption(_translate_text("Add a YouTube URL to a transcript to embed the player here."))
                else:
                    sel_row = st.selectbox(_translate_text("Jump to hit:"), range(len(kwic_df)), format_func=lambda i: f"[{kwic_df.iloc[i]['Timestamp']}] {kwic_df.iloc[i]['Node']} in {kwic_df.iloc[i]['Text']}")
                    r = kwic_df.iloc[sel_row]
                    secs = None
                    ts_str = r["Timestamp"]
                    if ts_str:
                        parts = ts_str.split(":")
                        secs = int(parts[-1]) + 60 * int(parts[-2]) + (3600 * int(parts[-3]) if len(parts) == 3 else 0)
                    st.video(f"https://www.youtube.com/embed/{yt}?start={int(secs or 0)}&autoplay=0")
                    st.caption(_translate_text(f"Context: ... {r['Left']} **{r['Node']}** {r['Right']} ..."))

# ============================================================
# TAB 3: CORPUS-GENERATED DICTIONARY + EDITABLE LEXICON
# ============================================================
with tab3:
    st.markdown(_translate_text("<h3 class='subheader'>Dictionary generated from uploaded texts</h3>", allow_fragments=True), unsafe_allow_html=True)
    st.caption(_translate_text("This corpus-derived word list updates from the texts stored in the corpus. It reports observed forms or lemmas, frequencies, and document coverage. It does not invent definitions; add definitions manually in your research notes or lexicon."))

    if not transcripts:
        st.info(_translate_text("Upload texts first to generate a corpus dictionary."))
    else:
        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc1:
            dict_lang = st.selectbox(_translate_text("Dictionary language"), ["all", "pl", "en", "mixed"],
                format_func=lambda x: {"all": "All transcript languages", "pl": "Polish", "en": "English", "mixed": "Mixed Polish-English"}[x], key="dict_language")
        with dc2:
            dict_unit = st.selectbox(_translate_text("Entry type"), ["lemma", "word"], key="dict_unit")
        with dc3:
            min_dict_freq = st.number_input(_translate_text("Minimum frequency"), min_value=1, max_value=1000, value=2, step=1, key="dict_min_freq")

        include_common = st.checkbox(_translate_text("Include common function words"), value=False, key="dict_common")
        dict_query = st.text_input(_translate_text("Filter entries"), placeholder="Type a word or part of a word", key="dict_query").strip().casefold()
        term_counts = Counter()
        term_docs = Counter()
        lang_counts = Counter()
        source_col = "lemma_counts" if dict_unit == "lemma" else "word_counts"
        for transcript in transcripts:
            lang_value = (transcript.get("language") or "pl").lower()
            if "-" in lang_value:
                transcript_lang = "mixed"
            elif lang_value.startswith("en"):
                transcript_lang = "en"
            else:
                transcript_lang = "pl"
            if dict_lang != "all" and transcript_lang != dict_lang:
                continue
            counts = transcript.get(source_col) or {}
            for term, count in counts.items():
                term = str(term).casefold().strip()
                count = int(count or 0)
                # Keep lexical items, excluding punctuation and numeric-only tokens.
                if not term or not re.search(r"[^\W\d_]", term, flags=re.UNICODE):
                    continue
                if not include_common:
                    stop = POLISH_STOP if transcript_lang == "pl" else set()
                    if transcript_lang == "en":
                        stop = {"the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at", "for", "from", "by", "with", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "my", "your", "his", "her", "our", "their", "not", "do", "does", "did", "have", "has", "had", "as", "so", "just", "very", "can", "could", "will", "would", "should", "about", "what", "which", "who", "when", "where", "how"}
                    if term in stop:
                        continue
                if dict_query and dict_query not in term:
                    continue
                term_counts[term] += count
                term_docs[term] += 1
                lang_counts[(term, transcript_lang)] += count
        dict_rows = []
        total_dict_tokens = sum(term_counts.values())
        for term, count in term_counts.most_common():
            if count < int(min_dict_freq):
                continue
            language_label = max(("pl", "en", "mixed"), key=lambda l: lang_counts[(term, l)])
            dict_rows.append({
                "Entry": term,
                "Language": language_label,
                "Frequency": count,
                "Texts": term_docs[term],
                "Per million": round(count / total_dict_tokens * 1_000_000, 2) if total_dict_tokens else 0,
            })
        dict_df = pd.DataFrame(dict_rows)
        if dict_df.empty:
            st.info(_translate_text("No entries match these filters. Lower the minimum frequency or include common words."))
        else:
            dict_top = st.slider("Maximum dictionary entries shown", min_value=25, max_value=1000, value=200, step=25, key="dict_limit")
            shown_dict = dict_df.head(dict_top)
            st.metric(_translate_text("Dictionary entries"), f"{len(dict_df):,}")
            st.dataframe(shown_dict, use_container_width=True, hide_index=True)
            st.download_button(_translate_text("Download corpus dictionary (CSV)"), dict_df.to_csv(index=False), file_name="corpus_dictionary.csv", mime="text/csv")

            selected_term = st.selectbox(_translate_text("Show concordance examples for an entry"), shown_dict["Entry"].tolist(), key="dict_example_term")
            if st.button(_translate_text("Show examples"), key="dict_show_examples"):
                match_col = "lemma" if dict_unit == "lemma" else "word"
                examples_res = (supabase.table("tokens").select("transcript_id,token_index,word,lemma")
                                .eq("corpus_id", corpus_id).eq(match_col, selected_term)
                                .order("transcript_id").order("token_index").limit(8).execute())
                examples = examples_res.data or []
                example_rows = []
                title_by_id = {t["id"]: t["title"] for t in transcripts}
                for hit in examples:
                    idx = int(hit["token_index"])
                    context = (supabase.table("tokens").select("token_index,word")
                               .eq("transcript_id", hit["transcript_id"])
                               .gte("token_index", max(0, idx - 5)).lte("token_index", idx + 5)
                               .order("token_index").execute()).data or []
                    left = " ".join(x["word"] for x in context if int(x["token_index"]) < idx)
                    right = " ".join(x["word"] for x in context if int(x["token_index"]) > idx)
                    example_rows.append({"Text": title_by_id.get(hit["transcript_id"], ""), "Left context": left,
                                         "Entry": hit["word"], "Right context": right})
                if example_rows:
                    st.dataframe(pd.DataFrame(example_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption(_translate_text("No indexed contexts are available for this entry yet."))

    st.divider()
    st.markdown(_translate_text("<h3 class='subheader'>Research lexicon: sentiment, modality, and stance</h3>", allow_fragments=True), unsafe_allow_html=True)
    st.caption(_translate_text("The corpus-generated dictionary above is built from your uploaded texts. This editable lexicon stores manually classified terms used by the stance and sentiment filters."))
    lex = cached_lexicon()
    lex_df = pd.DataFrame(lex)[["term", "lang", "category", "value"]] if lex else pd.DataFrame(columns=["term", "lang", "category", "value"])
    st.dataframe(lex_df, use_container_width=True, hide_index=True)
    with st.expander(_translate_text("Add dictionary entry")):
        n_term = st.text_input(_translate_text("Term / phrase"), key="lex_term")
        n_lang = st.selectbox(_translate_text("Language"), ["pl", "en"], key="lex_lang")
        n_cat = st.selectbox(_translate_text("Category"), ["positive", "negative", "epistemic", "deontic", "stance_agree", "stance_disagree", "custom"], key="lex_cat")
        n_val = st.number_input(_translate_text("Value (for sentiment: -1..1)"), -1.0, 1.0, 1.0, 0.5, key="lex_val")
        if st.button(_translate_text("Add entry"), key="lex_add"):
            if n_term.strip():
                db_insert("lexicon", {"term": n_term.strip().lower(), "lang": n_lang, "category": n_cat, "value": float(n_val)})
                clear_caches()
                st.rerun()
    del_opts = [f"{r['term']} [{r['lang']}/{r['category']}]" for _, r in lex_df.iterrows()] if not lex_df.empty else []
    if del_opts and st.multiselect(_translate_text("Delete entries"), del_opts, key="lex_del"):
        if st.button(_translate_text("Confirm delete"), key="lex_del_go"):
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
    st.markdown(_translate_text("<h3 class='subheader'>Frequency & time-sliced trends</h3>", allow_fragments=True), unsafe_allow_html=True)
    if not transcripts:
        st.info(_translate_text("Upload texts first."))
    else:
        f_mode = st.radio(_translate_text("Level:"), ["word", "lemma"], horizontal=True, key="freq_mode")
        gran = st.selectbox(_translate_text("Time slice granularity"), ["Year", "Quarter", "Month"], key="freq_gran")
        freq_df_all = corpus_word_freq(corpus_id, f_mode)
        top_terms = [w for w, _ in freq_df_all.most_common(40)]
        chosen = st.multiselect(_translate_text("Terms to track over time (empty = top 10 overall)"), top_terms, default=[], key="freq_terms")
        if not chosen:
            chosen = [w for w, _ in freq_df_all.most_common(10)]
        sliced = slice_transcripts(transcripts, gran)
        if sliced.empty:
            st.warning(_translate_text("No publish_date values set. Fill the mandatory 'Publish date' metadata to enable trends."))
        else:
            series = {s: Counter() for s in sorted(sliced["slice"].unique())}
            sizes = {s: 0 for s in series}
            col_name = "lemma_counts" if f_mode == "lemma" else "word_counts"
            for _, r in sliced.iterrows():
                for k, v in (r[col_name] or {}).items():
                    series[r["slice"]][k] += int(v)
                    sizes[r["slice"]] += int(v)
            trend_rows = []
            for slice_label in sorted(series):
                for term in chosen:
                    c = int(series[slice_label].get(term, 0))
                    trend_rows.append({"slice": str(slice_label), "term": str(term), "count": c,
                                       "per_million": float(round(c / sizes[slice_label] * 1e6, 2)) if sizes[slice_label] else 0.0})
            tdf = pd.DataFrame(trend_rows, columns=["slice", "term", "count", "per_million"])
            if tdf.empty or not chosen or not tdf["per_million"].notna().any():
                st.warning(_translate_text("No trend points can be plotted yet. Add dated transcripts containing searchable words, or change the selected terms."))
            else:
                tdf["per_million"] = pd.to_numeric(tdf["per_million"], errors="coerce")
                try:
                    fig = px.line(tdf, x="slice", y="per_million", color="term", markers=True,
                                  title=f"Frequency per million words ({gran.lower()}ly)")
                    safe_plotly_chart(fig, use_container_width=True)
                except (ValueError, TypeError, KeyError) as e:
                    st.warning(_translate_text(f"Could not draw the trend chart for this selection: {e}. The frequency table below remains available."))
                st.dataframe(tdf.pivot(index="slice", columns="term", values="count").fillna(0), use_container_width=True)

        st.divider()
        n_show = st.slider("Show top N", 10, 100, 25, key="freq_top")
        freq_df = pd.DataFrame(freq_df_all.most_common(n_show), columns=[f_mode.capitalize(), "Frequency"])
        freq_df["Rel. freq (per 1M)"] = (freq_df["Frequency"] / total_tokens * 1e6).round(1) if total_tokens else 0
        if total_tokens:
            safe_plotly_chart(lambda: make_bar_figure(freq_df.head(20), x=f_mode.capitalize(), y="Frequency", title=f"Most frequent {f_mode}s"), use_container_width=True)
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        st.download_button(_translate_text("Download full frequency list (CSV)"), pd.DataFrame(freq_df_all.most_common(), columns=[f_mode, "freq"]).to_csv(index=False), file_name="frequencies.csv", mime="text/csv")
# ============================================================
# TAB 5: NEOLOGISMS & KEYNESS (reference corpus comparison)
# ============================================================
with tab5:
    st.markdown(_translate_text("<h3 class='subheader'>Neologism / emergence spotter & keyness</h3>", allow_fragments=True), unsafe_allow_html=True)
    corpora_all = cached_corpora()
    other_opts = {c["id"]: c["name"] for c in corpora_all if c["id"] != corpus_id}
    if not other_opts:
        st.info(_translate_text("Create a second corpus (e.g. your small test corpus or a general reference corpus) to enable neologism spotting and keyness."))
    else:
        ref_id = st.selectbox(_translate_text("Reference corpus (baseline)"), list(other_opts), format_func=lambda i: other_opts[i], key="key_ref")
        gran_neo = st.selectbox(_translate_text("Time slice for emergence"), ["Year", "Quarter", "Month"], key="neo_gran")
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
            st.markdown(_translate_text("#### Emergent words (late vs early period)", allow_fragments=True))
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
                st.caption(_translate_text("Needs publish_date metadata on at least two time slices."))
        with c2:
            st.markdown(_translate_text("#### Keyness (log-likelihood, this corpus vs reference)", allow_fragments=True))
            min_freq = st.number_input(_translate_text("Min frequency in target corpus"), 1, 50, 5, key="key_min")
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
                st.download_button(_translate_text("Download keyness (CSV)"), key_df.to_csv(index=False), file_name="keyness.csv", mime="text/csv")
            else:
                st.caption(_translate_text("No terms reached LL >= 6.63 (p < 0.01). Lower the min frequency."))

# ============================================================
# TAB 6: SEMANTIC DRIFT & KEYWORDS IN TIME
# ============================================================
with tab6:
    st.markdown(_translate_text("<h3 class='subheader'>Semantic drift & keyword-in-time tracker</h3>", allow_fragments=True), unsafe_allow_html=True)
    gran_d = st.selectbox(_translate_text("Granularity"), ["Year", "Quarter", "Month"], key="drift_gran")
    sliced_d = slice_transcripts(transcripts, gran_d)
    if sliced_d.empty:
        st.warning(_translate_text("Set publish_date metadata to use time tracking."))
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

        st.markdown(_translate_text("#### Keywords in time (distinctive terms per slice)", allow_fragments=True))
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
        st.markdown(_translate_text("#### Semantic drift: collocates of a word over time", allow_fragments=True))
        drift_word = st.text_input(_translate_text("Word to track collocates for:"), placeholder="e.g. europa", key="drift_word")
        span = st.slider("Collocate span (words each side)", 2, 10, 5, key="drift_span")
        if drift_word and st.button(_translate_text("Compute drift"), key="drift_go"):
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
            st.caption(_translate_text("Compare collocates across slices: changing collocates = semantic drift (e.g. 'europa' drifting from 'unia' to 'kryzys')."))
            # heatmap
            top_coll = Counter()
            for c in drift_data.values():
                top_coll.update(c)
            top10 = [w for w, _ in top_coll.most_common(10)]
            if top10 and sl:
                hm = pd.DataFrame({s: {w: drift_data[s].get(w, 0) for w in top10} for s in sl})
                safe_plotly_chart(lambda: px.imshow(hm, text_auto=True, aspect="auto", title=f"Collocates of '{drift_word}' across time slices"), use_container_width=True)
            else:
                st.info(_translate_text(f"No collocates found for '{drift_word}' in this corpus. Check the spelling (the search is case-sensitive and matches the stored lowercase form)."))

# ============================================================
# TAB 7: LOANWORDS & ANGLICISMS
# ============================================================
with tab7:
    st.markdown(_translate_text("<h3 class='subheader'>Loanword & anglicism tracker (Polish)</h3>", allow_fragments=True), unsafe_allow_html=True)
    ang = cached_anglicisms()
    ang_df = pd.DataFrame(ang)[["term", "category"]] if ang else pd.DataFrame(columns=["term", "category"])
    st.caption(_translate_text("A stored list of English-origin items used in Polish. Extend it below; it persists for everyone."))
    with st.expander(_translate_text("Add anglicism")):
        a_term = st.text_input(_translate_text("Term"), key="ang_term")
        a_cat = st.text_input(_translate_text("Domain (e.g. tech, social media)"), key="ang_cat")
        if st.button(_translate_text("Add"), key="ang_add") and a_term.strip():
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
        st.markdown(_translate_text(f"#### {len(shown)} anglicisms found in this corpus", allow_fragments=True))
        if not shown.empty:
            safe_plotly_chart(lambda: make_bar_figure(shown.head(20), x="Term", y="Count in corpus", color="Domain", title="Anglicisms by frequency"), use_container_width=True)
        else:
            st.info(_translate_text("None of the anglicisms from the list occurs in this corpus yet."))
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button(_translate_text("Download anglicisms (CSV)"), ang_counts.to_csv(index=False), file_name="anglicisms.csv", mime="text/csv")
        # per-slice trend of top anglicisms
        gran_a = st.selectbox(_translate_text("Time granularity for anglicism trends"), ["Year", "Quarter", "Month"], key="ang_gran")
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
            if a_rows:
                safe_plotly_chart(lambda: px.line(pd.DataFrame(a_rows), x="slice", y="per_million", color="term", markers=True, title="Top anglicisms over time"), use_container_width=True)

    st.divider()
    st.markdown(_translate_text("#### Possible English-origin candidates from your Polish and English texts", allow_fragments=True))
    st.caption(_translate_text("These are frequency-overlap candidates for manual review, not confirmed loanwords. Shared forms can be cognates, names, abbreviations, or coincidental matches."))
    min_pl_freq = st.number_input(_translate_text("Minimum frequency in Polish texts"), min_value=1, max_value=100, value=2, key="ang_overlap_min")
    polish_freq, english_freq = Counter(), Counter()
    for tr in transcripts:
        lang = (tr.get("language") or "pl").lower()
        wc = tr.get("word_counts") or {}
        if lang == "pl":
            polish_freq.update({str(k).casefold(): int(v) for k, v in wc.items()})
        elif lang == "en":
            english_freq.update({str(k).casefold(): int(v) for k, v in wc.items()})
    known_ang = {a["term"].casefold() for a in ang}
    eng_stop = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "is", "are", "was", "were", "be", "it", "this", "that", "you", "we", "they", "i", "he", "she", "for", "with", "as", "at", "by", "from"}
    overlap_rows = []
    for term in (polish_freq.keys() & english_freq.keys()):
        if (len(term) < 3 or term in eng_stop or term in POLISH_STOP or term in known_ang
                or not re.search(r"[^\W\d_]", term, flags=re.UNICODE)
                or polish_freq[term] < int(min_pl_freq) or english_freq[term] < 2):
            continue
        overlap_rows.append({"Possible candidate": term, "Polish frequency": polish_freq[term], "English frequency": english_freq[term]})
    overlap_df = pd.DataFrame(overlap_rows).sort_values("Polish frequency", ascending=False) if overlap_rows else pd.DataFrame()
    if overlap_df.empty:
        st.info(_translate_text("No candidates yet. This comparison needs at least one pure Polish transcript and one pure English transcript in the same corpus."))
    else:
        st.dataframe(overlap_df.head(200), use_container_width=True, hide_index=True)
        candidates_to_add = st.multiselect(_translate_text("Choose candidates to add to the anglicism list"), overlap_df["Possible candidate"].head(200).tolist(), key="ang_candidates_add")
        if st.button(_translate_text("Add selected candidates to anglicism list"), key="ang_candidates_save"):
            for term in candidates_to_add:
                db_insert("anglicisms", {"term": term, "category": "candidate, review needed"})
            clear_caches()
            st.success(_translate_text(f"Added {len(candidates_to_add)} candidate(s). Review them in the anglicism list before treating them as established loans."))
            st.rerun()

# ============================================================
# TAB 8: STANCE / SENTIMENT / MODALITY
# ============================================================
with tab8:
    st.markdown(_translate_text("<h3 class='subheader'>Stance, sentiment & modality filter</h3>", allow_fragments=True), unsafe_allow_html=True)
    lex = cached_lexicon()
    if not lex:
        st.info(_translate_text("Add terms in the Dictionaries tab first."))
    else:
        sent = {l["term"]: l["value"] for l in lex if l["lang"] == "pl" and l["category"] in ("positive", "negative")}
        epi = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "epistemic"}
        deo = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "deontic"}
        agr = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "stance_agree"}
        dis = {l["term"] for l in lex if l["lang"] == "pl" and l["category"] == "stance_disagree"}

        st.markdown(_translate_text("Filter transcripts by stance / sentiment / modality profile:", allow_fragments=True))
        c1, c2, c3 = st.columns(3)
        want_sent = c1.selectbox(_translate_text("Sentiment tilt"), ["any", "positive-lean", "negative-lean"], key="st_sent")
        want_epi = c2.slider("Min epistemic markers (hedging)", 0, 20, 0, key="st_epi")
        want_deo = c3.slider("Min deontic markers (obligation)", 0, 20, 0, key="st_deo")
        want_stance = c1.selectbox(_translate_text("Stance"), ["any", "agree-lean", "disagree-lean"], key="st_stance")

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
        st.caption(_translate_text("Counts come from the stored lexicon; edit it in the Dictionaries tab to refine the filter."))

# ============================================================
# TAB 9: STATISTICS & METADATA
# ============================================================
with tab9:
    st.markdown(_translate_text("<h3 class='subheader'>Corpus statistics</h3>", allow_fragments=True), unsafe_allow_html=True)
    meta_fields = cached_metadata_fields()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_translate_text("#### Tokens per text", allow_fragments=True))
        if transcripts:
            lengths = sorted([(t["title"], sum(int(v) for v in (t.get("word_counts") or {}).values())) for t in transcripts], key=lambda x: x[1])
            len_df = pd.DataFrame(lengths, columns=["text", "tokens"])
            if not len_df.empty and len_df["tokens"].sum() > 0:
                safe_plotly_chart(lambda: make_bar_figure(len_df, x="tokens", y="text", orientation="h", title="Tokens per text"), use_container_width=True)
            else:
                st.info(_translate_text("No token counts yet. Upload and index texts first."))
    with c2:
        st.markdown(_translate_text("#### Vocabulary richness", allow_fragments=True))
        lm = corpus_word_freq(corpus_id, "lemma")
        n_tok = sum(lm.values())
        if n_tok:
            st.metric(_translate_text("Tokens"), f"{n_tok:,}")
            st.metric(_translate_text("Types"), f"{len(lm):,}")
            st.metric(_translate_text("TTR"), f"{len(lm) / n_tok:.4f}")
            st.metric(_translate_text("Guiraud index"), f"{len(lm) / math.sqrt(n_tok):.2f}")

    st.divider()
    st.markdown(_translate_text("#### Metadata coverage", allow_fragments=True))
    if transcripts:
        cov_rows = []
        for f in meta_fields:
            filled = sum(1 for t in transcripts if t.get(f["field_name"]) not in (None, ""))
            cov_rows.append({"Field": f["label"], "Mandatory": "yes" if f["mandatory"] else "no", "Filled": f"{filled}/{len(transcripts)}"})
        st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
        df_all = pd.DataFrame([{f["field_name"]: t.get(f["field_name"]) for f in meta_fields} | {"title": t["title"]} for t in transcripts])
        st.download_button(_translate_text("Download metadata table (CSV)"), df_all.to_csv(index=False), file_name="metadata.csv", mime="text/csv")

    st.divider()
    st.markdown(_translate_text("#### POS distribution", allow_fragments=True))
    pos_total = Counter()
    for t in transcripts:
        pos_total.update({k: int(v) for k, v in (t.get("pos_counts") or {}).items()})
    if pos_total:
        pos_df = pd.DataFrame(pos_total.most_common(), columns=["POS", "Count"])
        if not pos_df.empty and pos_df["Count"].sum() > 0:
            safe_plotly_chart(lambda: px.pie(pos_df.head(12), values="Count", names="POS", title="POS distribution"), use_container_width=True)

with tab10:
    st.markdown(_translate_text("<h3 class='subheader'>Annotate a transcript using corpus resources</h3>", allow_fragments=True), unsafe_allow_html=True)
    st.caption(_translate_text("Annotations are created from the stored anglicism list, research lexicon, selected corpus words, and POS tags. The highlighted preview and exported DOCX are copies; your source transcript is unchanged."))
    if not transcripts:
        st.info(_translate_text("Upload transcripts first."))
    else:
        transcript_ids = [t["id"] for t in transcripts]
        transcript_by_id = {t["id"]: t for t in transcripts}
        picked_id = st.selectbox(_translate_text("Transcript to annotate"), transcript_ids,
                                 format_func=lambda i: f"{transcript_by_id[i]['title']} ({transcript_by_id[i].get('language') or 'pl'})",
                                 key="annotation_transcript")
        selected_transcript = transcript_by_id[picked_id]
        annotation_text = selected_transcript.get("clean_text") or selected_transcript.get("raw_text") or ""
        ann_lang = (selected_transcript.get("language") or "pl").lower()
        allowed_lex_languages = {"pl", "en"} if "-" in ann_lang else {"en" if ann_lang == "en" else "pl"}

        row_a, row_b = st.columns(2)
        with row_a:
            use_anglicisms = st.checkbox(_translate_text("Mark known anglicisms"), value=True, key="ann_anglicisms")
            use_lexicon = st.checkbox(_translate_text("Mark sentiment, modality, and stance entries"), value=True, key="ann_lexicon")
        with row_b:
            use_dictionary = st.checkbox(_translate_text("Mark selected corpus-dictionary words"), value=False, key="ann_dictionary")
            pos_choices = st.multiselect(_translate_text("Mark POS categories"), ["NOUN", "PROPN", "VERB", "ADJ", "ADV"], key="ann_pos")

        annotation_terms = []
        if use_anglicisms and (ann_lang.startswith("pl") or "-" in ann_lang):
            for entry in cached_anglicisms():
                tag = "Kandydat anglicyzmu" if "candidate" in str(entry.get("category", "")).casefold() else "Anglicyzm"
                annotation_terms.append((entry["term"], tag))
        if use_lexicon:
            lex_categories = st.multiselect(
                _translate_text("Lexicon categories"), ["positive", "negative", "epistemic", "deontic", "stance_agree", "stance_disagree", "custom"],
                default=["positive", "negative", "epistemic", "deontic", "stance_agree", "stance_disagree"], key="ann_lex_categories")
            lex_label = {"positive": "Sentyment pozytywny", "negative": "Sentyment negatywny",
                         "epistemic": "Modalność epistemiczna", "deontic": "Modalność deontyczna",
                         "stance_agree": "Stanowisko: zgoda", "stance_disagree": "Stanowisko: niezgoda", "custom": "Termin własny"}
            for entry in cached_lexicon():
                if entry.get("lang", "pl") in allowed_lex_languages and entry.get("category") in lex_categories:
                    annotation_terms.append((entry["term"], lex_label.get(entry["category"], "Termin własny")))
        if use_dictionary:
            dictionary_options = []
            for word, count in Counter(selected_transcript.get("word_counts") or {}).most_common(500):
                term = str(word).casefold()
                if re.search(r"[^\W\d_]", term, flags=re.UNICODE) and term not in POLISH_STOP:
                    dictionary_options.append(term)
            chosen_dictionary_words = st.multiselect(_translate_text("Corpus terms to highlight"), dictionary_options[:300], key="ann_dict_terms")
            annotation_terms.extend((word, "Hasło korpusowe") for word in chosen_dictionary_words)
        if pos_choices:
            pos_tokens = fetch_all("tokens", select="word,pos", eq={"transcript_id": picked_id})
            pos_labels = {p: f"Część mowy: {p}" for p in pos_choices}
            for tok in pos_tokens:
                if tok.get("pos") in pos_choices:
                    annotation_terms.append((tok.get("word", ""), pos_labels[tok["pos"]]))

        if st.button(_translate_text("Generate annotations"), type="primary", key="ann_generate"):
            spans = find_annotation_spans(annotation_text, annotation_terms)
            st.session_state["annotation_result"] = {"transcript_id": picked_id, "text": annotation_text, "spans": spans}
        result = st.session_state.get("annotation_result")
        if result and result.get("transcript_id") == picked_id:
            spans = result["spans"]
            if not spans:
                st.info(_translate_text("No matches found for the selected annotation layers."))
            else:
                st.metric(_translate_text("Annotated occurrences"), len(spans))
                st.markdown(annotation_html(annotation_text, spans), unsafe_allow_html=True)
                spans_df = pd.DataFrame(spans)[["start", "end", "term", "label"]]
                st.dataframe(spans_df, use_container_width=True, hide_index=True)
                st.download_button(_translate_text("Download annotation spans (CSV)"), spans_df.to_csv(index=False),
                                   file_name="annotation_spans.csv", mime="text/csv")
                safe_title = re.sub(r"[^\w-]+", "_", selected_transcript["title"])
                annotated_docx = annotation_docx_bytes(annotation_text, spans, selected_transcript["title"])
                st.download_button(_translate_text("Download annotated Word document"), annotated_docx,
                                   file_name=f"{safe_title}_annotated.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

st.divider()
st.caption(_translate_text("Linguistic Corpus Engine v3.0 | Streamlit + spaCy + Supabase. Data persists in the cloud; anyone with the app link can browse, search and analyze. Editing/deleting is available to all viewers; use Supabase RLS if you need to restrict this."))
