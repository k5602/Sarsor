import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
from googletrans import Translator
from textblob import TextBlob
import requests
import langdetect
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from camel_tools.utils.normalize import normalize_unicode
import fitz
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import rake_nltk
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re as regex
import os

nltk.data.path.append("/home/appuser/nltk_data")
os.makedirs("/home/appuser/nltk_data", exist_ok=True)
nltk.download('punkt', quiet=True, download_dir="/home/appuser/nltk_data")
nltk.download('stopwords', quiet=True, download_dir="/home/appuser/nltk_data")
nltk.download('punkt_tab', quiet=True, download_dir="/home/appuser/nltk_data")  # Critical fix
nltk.download('wordnet', quiet=True, download_dir="/home/appuser/nltk_data")
nltk.download('omw-1.4', quiet=True, download_dir="/home/appuser/nltk_data")  # For lemmatization
translator = Translator()

def arabic_sentence_tokenize(text):
    endings = ['؟', '!', '.', '؛', '\n', '\r\n']
    pattern = '|'.join(map(re.escape, endings))
    sentences = [s.strip() for s in re.split(f'(?<=[{pattern}])', text) if s.strip()]
    return sentences

def arabic_word_tokenize(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', text)
    return words

def preprocess_text(text, language='ar'):
    if language == 'ar':
        text = re.sub(r'[^\u0600-\u06FF\s\.\!\؟\،\؛]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hybrid_tfidf_textrank_summarize(text, num_sentences=3, language='ar'):
    if not text:
        return ""
    
    try:
        text = preprocess_text(text, language)
        
        if language == 'ar':
            sentences = arabic_sentence_tokenize(text)
            try:
                stop_words = list(stopwords.words('arabic'))
            except:
                stop_words = []
        else:
            sentences = sent_tokenize(text)
            try:
                stop_words = list(stopwords.words('english'))
            except:
                stop_words = []

        if len(sentences) < 2:
            return text
                
        vectorizer = TfidfVectorizer(
            stop_words=stop_words, 
            use_idf=True, 
            smooth_idf=True, 
            sublinear_tf=True, 
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        try:
            scores = nx.pagerank(nx_graph, max_iter=200)
        except:
            scores = nx.degree_centrality(nx_graph)
        
        tfidf_scores = tfidf_matrix.sum(axis=1).A1
        
        combined_scores = {}
        for i in range(len(sentences)):
            combined_scores[i] = 0.6 * tfidf_scores[i] + 0.4 * scores[i]
        
        num_sentences = min(num_sentences, len(sentences))
        ranked_sentences = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
        selected_indices.sort()
        
        return ' '.join([sentences[i] for i in selected_indices])
        
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return ' '.join(sentences[:min(num_sentences, len(sentences))])

def analyze_sentiment(text, language='ar'):
    if language == 'ar':
        positive_words = set(["جيد", "رائع", "ممتاز", "صحيح", "مفيد", "مهم"])
        negative_words = set(["سيء", "سيئ", "خاطئ", "غير صحيح", "غير مفيد", "غير مهم"])
        words = arabic_word_tokenize(text)
        pos_count = sum(1 for word in words if word.lower() in positive_words)
        neg_count = sum(1 for word in words if word.lower() in negative_words)
        total_words = len(words)
        if total_words == 0:
            return 0.0, 0.0
        polarity = (pos_count - neg_count) / total_words
        subjectivity = (pos_count + neg_count) / total_words
        return polarity, subjectivity
    else:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity

def extract_combined_rake_tfidf_keywords(text, num_keywords=5, language='ar'):
    if not text:
        return []
    text = preprocess_text(text, language)
    words = arabic_word_tokenize(text) if language == 'ar' else word_tokenize(text)
    try:
        stop_words = list(stopwords.words('arabic')) if language == 'ar' else list(stopwords.words('english'))
    except:
        stop_words = []
    words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    rake = rake_nltk.Rake(stopwords=stop_words)
    rake.extract_keywords_from_text(text)
    rake_keywords = rake.get_ranked_phrases_with_scores()[:num_keywords]
    vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=True, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    tfidf_keywords = [(tfidf_scores[i], feature_names[i]) for i in tfidf_scores.argsort()[-num_keywords:][::-1]]
    combined_keywords = {}
    for score, keyword in rake_keywords:
        combined_keywords[keyword] = score
    for score, keyword in tfidf_keywords:
        if keyword in combined_keywords:
            combined_keywords[keyword] = (combined_keywords[keyword] + score) / 2
        else:
            combined_keywords[keyword] = score
    sorted_keywords = sorted(combined_keywords.items(), key=lambda x: (-x[1], len(x[0])))
    filtered_keywords = [keyword for keyword, score in sorted_keywords if 3 <= len(keyword.split()) <= 5]
    return filtered_keywords[:num_keywords]

def identify_hard_words(text, language='ar'):
    if language == 'ar':
        detected_language = langdetect.detect(text)
        if detected_language != 'ar':
            text = translate_text(text, src=detected_language, dest='ar')
        words = arabic_word_tokenize(text)
        try:
            stop_words = list(stopwords.words('arabic'))
        except:
            stop_words = []
    else:
        words = word_tokenize(text)
        try:
            stop_words = list(stopwords.words('english'))
        except:
            stop_words = []
    hard_words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    unique_hard_words = list(set(hard_words))
    hard_word_info = {}
    for word in unique_hard_words:
        translated_word = translate_text(word, src='ar' if language == 'ar' else 'en', 
                                      dest='en' if language == 'ar' else 'ar')
        example_sentence = fetch_example_sentence(word, language)
        hard_word_info[word] = {
            'translation': translated_word,
            'example': example_sentence
        }
    return hard_word_info

def fetch_example_sentence(term, language='ar'):
    if language == 'ar':
        term_en = translator.translate(term, src='ar', dest='en').text
    else:
        term_en = term
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term_en}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                meanings = data[0].get('meanings', [])
                for meaning in meanings:
                    examples = meaning.get('definitions', [{}])[0].get('example', None)
                    if examples:
                        return examples
    except Exception as e:
        st.error(f"Error fetching example sentence: {str(e)}")
    return f"No example found for {term}"

def translate_text(text, src='ar', dest='en'):
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return translated.text
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return text

def fetch_website_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
            content_elements = []
            for tag in content_tags:
                content_elements.extend(soup.find_all(tag))
            text = ' '.join([element.get_text(strip=True) for element in content_elements])
            return text
        else:
            st.error(f"Failed to fetch content from {url}")
    except Exception as e:
        st.error(f"Error fetching website content: {str(e)}")
    return ""

def read_pdf(file_stream):
    try:
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def ocr_image(image_stream):
    try:
        image = Image.open(image_stream)
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        st.error(f"Error performing OCR: {str(e)}")
        return ""

def named_entity_recognition_rule_based(text, language='ar'):
    if language == 'ar':
        tokens = arabic_word_tokenize(text)
        named_entities = []
        for token in tokens:
            if token[0].isupper():
                named_entities.append((token, 'PERSON'))
        dates = regex.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', text)
        named_entities.extend([(date, 'DATE') for date in dates])
        organizations = regex.findall(r'\b(?:شركة|مؤسسة|جمعية|منظمة)\s+[^\s]+(?:\s+[^\s]+)*\b', text)
        named_entities.extend([(org, 'ORGANIZATION') for org in organizations])
        return named_entities
    else:
        tokens = word_tokenize(text)
        named_entities = []
        for token in tokens:
            if token[0].isupper() and len(token) > 3:
                named_entities.append((token, 'PERSON'))
        dates = regex.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', text)
        named_entities.extend([(date, 'DATE') for date in dates])
        organizations = regex.findall(r'\b(?:Company|Organization|Institute|Foundation)\s+[^\s]+(?:\s+[^\s]+)*\b', text)
        named_entities.extend([(org, 'ORGANIZATION') for org in organizations])
        return named_entities

def create_glossary(text, language='ar'):
    hard_words_info = identify_hard_words(text, language)
    glossary = []
    for word, info in hard_words_info.items():
        glossary.append((word, info['translation'], info['example']))
    return glossary

st.title("ٍSarsor: Arabic Text Analysis and Summarization(0.01)")

st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] p {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

input_option = st.selectbox("Input Method", ["Upload File (PDF/Image)", "Paste Text", "Enter Website URL"])

if input_option == "Upload File (PDF/Image)":
    uploaded_file = st.file_uploader("Choose a PDF or Image file", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name.endswith('.pdf'):
            text = read_pdf(uploaded_file)
        elif file_name.endswith(('.jpg', '.jpeg', '.png')):
            text = ocr_image(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF or an image.")
            text = None
    else:
        text = None
elif input_option == "Paste Text":
    text = st.text_area("Paste text here", height=200)
else:
    url = st.text_input("Enter Website URL (e.g., Wikipedia page)")
    if url:
        text = fetch_website_content(url)
    else:
        text = None

if st.button("Summarize", key="summarize_button"):
    if not text:
        st.warning("Please provide some text to summarize.")
    else:
        try:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            detected_language = langdetect.detect(text)
            language_code = 'ar' if detected_language == 'ar' else 'en'
            progress_bar.progress(10)
            progress_text.text("Detecting language...")
            summary = hybrid_tfidf_textrank_summarize(text, num_sentences=3, language=language_code)
            progress_bar.progress(30)
            progress_text.text("Summarizing text...")
            st.subheader("Summary")
            st.text(summary)
            polarity, subjectivity = analyze_sentiment(text, language=language_code)
            progress_bar.progress(50)
            progress_text.text("Analyzing sentiment...")
            st.subheader("Sentiment Analysis")
            st.write(f"Polarity: {polarity} (Range: -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)")
            st.write(f"Subjectivity: {subjectivity} (Range: 0 to 1, where 0 is objective and 1 is subjective)")
            keywords = extract_combined_rake_tfidf_keywords(text, num_keywords=5, language=language_code)
            progress_bar.progress(70)
            progress_text.text("Extracting keywords...")
            st.subheader("Keywords")
            st.write(", ".join(keywords))
            named_entities = named_entity_recognition_rule_based(text, language=language_code)
            if named_entities:
                st.subheader("Named Entities")
                for entity, label in named_entities:
                    st.write(f"Entity: {entity}, Label: {label}")
            else:
                st.write("No named entities found.")
            progress_bar.progress(80)
            progress_text.text("Identifying named entities...")
            if st.checkbox("Translate Summary to Other Language", key="translate_checkbox"):
                translated_summary = translate_text(summary, src=language_code, dest='en' if language_code == 'ar' else 'ar')
                st.subheader("Translated Summary")
                st.text(translated_summary)
            progress_bar.progress(90)
            progress_text.text("Translating summary...")
            glossary = create_glossary(text, language_code)
            if glossary:
                st.subheader("Glossary of Hard Words")
                for word, translation, example in glossary:
                    st.write(f"**{word} ({translation})**: {example}")
            progress_bar.progress(100)
            progress_text.text("Completed!")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please provide some text to summarize.")
