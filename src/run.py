import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import re
from nltk.corpus import stopwords

st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎥", layout="centered")

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model.pkl")
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please make sure 'sentiment_model.pkl' is in the 'models' folder.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ----------------- Load Stopwords -----------------
try:
    stop_words = set(stopwords.words("english"))
except:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ----------------- Custom CSS -----------------
st.markdown("""
<style>
.word-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
    padding: 10px;
    background-color: #1a1d23;
    border-radius: 10px;
}
.word-badge {
    padding: 6px 10px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 500;
    color: white;
}
.stopword {
    background-color: rgba(231, 76, 60, 0.3);
    border: 1px solid #e74c3c;
}
.keptword {
    background-color: rgba(46, 204, 113, 0.3);
    border: 1px solid #2ecc71;
}
.result-box {
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
}
.positive {background: rgba(46, 204, 113, 0.15); border: 1px solid #2ecc71; color: #2ecc71;}
.negative {background: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; color: #e74c3c;}
.neutral {background: rgba(241, 196, 15, 0.15); border: 1px solid #f1c40f; color: #f1c40f;}
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Function -----------------
def analyze_text_with_stopwords(text):
    # Tokenize
    tokens = re.findall(r"\b\w+\b", text.lower())
    removed = [w for w in tokens if w in stop_words]
    kept = [w for w in tokens if w not in stop_words]

    # Predict sentiment
    prediction = model.predict([text])[0]
    probas = model.predict_proba([text])[0]
    confidence = round(max(probas) * 100, 2)
    label = prediction.capitalize()

    return label, confidence, removed, kept

st.markdown("## 🎥 **Movie Review Sentiment Analyzer**")
st.write("Type your review below or upload a text file to analyze multiple reviews.")

review = st.text_area("✍️ Write your review here:", placeholder="e.g., The movie was absolutely stunning!")
uploaded_file = st.file_uploader("📁 Or upload a .txt or .csv file with reviews", type=["txt", "csv"])

# ----------------- Main Logic -----------------
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "" and uploaded_file is None:
        st.warning("Please enter a review or upload a file!")
    else:
        results = []

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "review" not in df.columns:
                    st.error("CSV must contain a 'review' column.")
                    st.stop()
                texts = df["review"].dropna().tolist()
            else:
                texts = uploaded_file.read().decode("utf-8").splitlines()
        else:
            texts = [review]

        for text in texts:
            sentiment, confidence, removed, kept = analyze_text_with_stopwords(text)
            results.append({
                "Review": text[:100] + "...",
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Removed": removed,
                "Kept": kept
            })

        result_df = pd.DataFrame(results)

        # --- Sentiment Result Card ---
        sentiment = result_df.iloc[0]["Sentiment"]
        confidence = result_df.iloc[0]["Confidence"]

        if sentiment == "Positive":
            st.markdown(f"<div class='result-box positive'>✅ <b>Positive Review</b><br>{confidence}% confidence</div>", unsafe_allow_html=True)
        elif sentiment == "Negative":
            st.markdown(f"<div class='result-box negative'>❌ <b>Negative Review</b><br>{confidence}% confidence</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box neutral'>😐 <b>Neutral Review</b><br>{confidence}% confidence</div>", unsafe_allow_html=True)

# --- Confidence Breakdown for Single Review ---
        probas = model.predict_proba([review])[0]
        sentiment_labels = model.classes_

        st.subheader("📊 Confidence Breakdown")
        conf_df = pd.DataFrame({
            "Sentiment": sentiment_labels,
            "Confidence": [round(p * 100, 2) for p in probas]
        })

        conf_chart = px.bar(
            conf_df,
            x="Sentiment",
            y="Confidence",
            color="Sentiment",
            title="Model Confidence per Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#f1c40f"
            }
        )
        st.plotly_chart(conf_chart, use_container_width=True)


        # --- Detailed Results ---
        st.subheader("🧾 Detailed Results")
        st.dataframe(result_df[["Review", "Sentiment", "Confidence"]], use_container_width=True)

        # --- Stopword Display for single review ---
        if len(result_df) == 1:
            removed = result_df.iloc[0]["Removed"]
            kept = result_df.iloc[0]["Kept"]

            st.subheader("🧹 Stopword Breakdown")
            st.markdown("**🟥 Removed Stopwords:**")
            if removed:
                st.markdown("<div class='word-container'>" + "".join(
                    [f"<span class='word-badge stopword'>{w}</span>" for w in removed]
                ) + "</div>", unsafe_allow_html=True)
            else:
                st.write("_No stopwords removed._")

            st.markdown("**🟩 Words Kept for Analysis:**")
            if kept:
                st.markdown("<div class='word-container'>" + "".join(
                    [f"<span class='word-badge keptword'>{w}</span>" for w in kept]
                ) + "</div>", unsafe_allow_html=True)
            else:
                st.write("_No words kept after cleaning._")

        
