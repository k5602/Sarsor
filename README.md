# 🕌 Sarsor: Arabic Text Analysis & Summarization Tool (v0.01)  
**Unlock the power of Arabic text with AI-driven analysis, summarization, and insights!**  

---

## 🌟 Features  
- **Hybrid Summarization**: Combines TF-IDF and TextRank for accurate Arabic/English summaries.  
- **Sentiment Analysis**: Detects polarity and subjectivity in Arabic/English text.  
- **Keyword Extraction**: RAKE + TF-IDF fusion for identifying key terms.  
- **Named Entity Recognition (NER)**: Rule-based detection of people, dates, and organizations.  
- **Multi-Input Support**:  
  - 📄 PDF/text extraction  
  - 🌐 Website content scraping  
  - 🖼️ OCR for images  
  - ✍️ Direct text input  
- **Translation**: Arabic ↔️ English summary translation.  
- **Glossary Builder**: Auto-generates hard-word definitions with examples.  
- **RTL Support**: Native Arabic text rendering.  

---

## 🚀 Quick Start  

### Installation  
1. **Install Dependencies**:  
   ```bash
   pip install streamlit nltk googletrans==4.0.0-1 textblob requests langdetect camel-tools pytesseract PyMuPDF beautifulsoup4 scikit-learn networkx rake-nltk
   ```
2. **Install Tesseract OCR** (for image processing):  
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)  
   - **MacOS**: `brew install tesseract`  
   - **Linux**: `sudo apt install tesseract-ocr`  

### Usage  
```bash
streamlit run Sarsor.py
```

---

## 🖥️ UI Walkthrough  
1. **Input Options**:  
   - Upload PDF/Image 📤  
   - Paste text ✍️  
   - Enter website URL 🌐  

2. **Outputs**:  
   - Summary with translation option 🌍  
   - Sentiment scores 📊  
   - Keywords 🔑  
   - Named Entities 🏷️  
   - Interactive glossary 📖  
---

## 🛠️ Tech Stack  
- **NLP**: `nltk`, `rake-nltk`, `TextBlob`, `langdetect`  
- **Translation**: `googletrans`  
- **OCR**: `pytesseract`, `PIL`  
- **PDF Processing**: `PyMuPDF`  
- **Web Scraping**: `BeautifulSoup`  
- **ML**: `scikit-learn`, `networkx`  
- **UI**: `streamlit`  

---

## 📚 Example Use Cases  
- Academic research paper summarization 📑  
- News article sentiment analysis 📰  
- Document keyword extraction for SEO 🔍  
- Language learning glossary generation 🎓  

---


## 🤝 Contributing  
Found a bug? Have an idea?  
1. Fork the repo  
2. Create a PR with your changes  
3. Follow the [Code of Conduct](https://www.contributor-covenant.org/)  

---

## 📜 License  
MIT License - Free for educational and commercial use.  

---

**Crafted with ❤️ by [Khaled]**  
--- 

[![Star on GitHub](https://img.shields.io/github/stars/k5602/Sarsor?style=social)](https://github.com/k5602/Sarsor)  
