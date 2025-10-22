# 🎬 Movie Sentiment Analyzer

An interactive **Streamlit web app** that analyzes the **sentiment of movie reviews** using a trained **Machine Learning model**.  
It classifies each review as **Positive**, **Negative**, or **Neutral**, and gives deep insights such as stopword breakdowns, model confidence, and word frequency analysis.

---

## 🚀 Features

### 🧩 Intelligent ML Model
- Built with **TF-IDF** vectorization and a **Logistic Regression** classifier.
- Trained on cleaned real-world text data using custom preprocessing.
- Delivers accurate predictions with confidence scores for each sentiment.

### 🎨 Interactive Streamlit Dashboard
- Clean, responsive interface designed with custom CSS.
- Supports both **single text input** and **bulk review uploads** (`.csv` or `.txt`).
- Displays real-time analytics and visual feedback.

### 📊 Visual Insights
- **Model Confidence Breakdown** – see how confident the AI is for each sentiment class.
- **Sentiment Distribution Chart** – shows sentiment trends across all uploaded reviews.
- **Confidence Histogram** – visualizes how confident predictions are across reviews.
- **Average Confidence per Sentiment** – helps identify model reliability by class.
- **Word Frequency & Length Charts** – reveal linguistic patterns in positive or negative texts.

### 🧹 NLP Preprocessing Visualization
- Shows which **stopwords were removed** and which **words were kept** during text cleaning.
- Words are displayed as colorful **badges** in a flexible container.
- Provides full transparency on how the model interprets input.

### 🛠️ Modular Code Design
- `preprocess.py` – Handles text cleaning and dataset preparation.  
- `train.py` – Trains and saves the sentiment analysis model.  
- `run.py` – Streamlit app for real-time predictions and visualization.  

---

## 🧰 Technologies Used

| Category | Tools |
|-----------|-------|
| **Language** | Python 3 |
| **Framework** | Streamlit |
| **ML Libraries** | Scikit-learn, Pandas, NLTK |
| **Visualization** | Plotly Express |
| **Model** | Logistic Regression + TF-IDF Vectorizer |

---

## 📁 Project Structure
```bash
movie-sentiment-analyzer/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── sentiment_model.pkl
│
├── src/
│   ├── __pycache__/
│   ├── preprocess.py
│   ├── train.py
│   └── run.py
│
├── .gitattributes
├── LICENSE
└── README.md

```

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/movie-sentiment-analyzer.git
   cd movie-sentiment-analyzer
   
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   
4. Run the code
   ```bash
   streamlit run run.py

