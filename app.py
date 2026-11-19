import streamlit as st
import pickle

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ==========================
# Custom CSS
# ==========================

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #1E3A8A;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
    margin-bottom: 20px;
}

.result-real {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
    color: #155724;
    font-size: 22px;
    font-weight: bold;
}

.result-fake {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 10px;
    color: #721c24;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==========================
# Header Section
# ==========================

st.markdown(
    "<div class='title'>📰 Fake News Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Logistic Regression + TF-IDF Machine Learning Model</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# Input Section
# ==========================

st.subheader("✍️ Enter News Article")

news = st.text_area(
    "",
    height=200,
    placeholder="Paste your news article here..."
)

# ==========================
# Predict Button
# ==========================

if st.button("🔍 Check News", use_container_width=True):

    if news.strip() == "":
        st.warning("⚠️ Please enter some news content first.")
    else:

        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        st.markdown("### 📊 Prediction Result")

        if prediction[0] == 1:
            st.markdown(
                "<div class='result-real'>✅ This appears to be REAL NEWS</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-fake'>❌ This appears to be FAKE NEWS</div>",
                unsafe_allow_html=True
            )

# ==========================
# Footer
# ==========================

st.markdown("---")

st.caption(
    "Developed using Logistic Regression, TF-IDF Vectorization and Streamlit"
)
