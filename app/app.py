import streamlit as st
import pandas as pd
import joblib
import re
import os
from pypdf import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "resume_category_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:

        st.error(
            f"Unable to read PDF: {e}"
        )

        return ""


# ============================================================
# PREDICTION
# ============================================================

def predict_resume_category(text, model, vectorizer):

    cleaned_text = clean_text(text)

    if not cleaned_text:

        return None, []

    # Convert text into TF-IDF features
    features = vectorizer.transform(
        [cleaned_text]
    )

    # Main prediction
    prediction = model.predict(
        features
    )[0]

    # Get probabilities if available
    top_predictions = []

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            features
        )[0]

        classes = model.classes_

        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        top_predictions = results[:3]

    return prediction, top_predictions


# ============================================================
# HEADER
# ============================================================

st.title("📄 AI Resume Parser")

st.markdown(
    """
    ### Automatically classify a resume using Machine Learning

    Upload a PDF resume and the AI model will predict the
    most likely professional category.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, vectorizer = load_model()

    st.success(
        "✅ AI model loaded successfully"
    )

except Exception as e:

    st.error(
        f"❌ Could not load the AI model: {e}"
    )

    st.info(
        "Make sure resume_category_model.pkl and "
        "tfidf_vectorizer.pkl are inside the app folder."
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📤 Upload Resume")

uploaded_file = st.file_uploader(
    "Choose a PDF resume",
    type=["pdf"]
)


# ============================================================
# PROCESS RESUME
# ============================================================

if uploaded_file is not None:

    st.success(
        f"📎 Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🔍 Analyze Resume",
        type="primary"
    ):

        with st.spinner(
            "Analyzing resume..."
        ):

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

        if not resume_text:

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.warning(
                "If this is a scanned/image-only PDF, "
                "OCR will be required."
            )

        else:

            st.success(
                "✅ Resume text extracted successfully!"
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction, top_predictions = (
                predict_resume_category(
                    resume_text,
                    model,
                    vectorizer
                )
            )

            if prediction is None:

                st.error(
                    "Unable to classify this resume."
                )

            else:

                st.divider()

                # ------------------------------------------------
                # MAIN RESULT
                # ------------------------------------------------

                st.subheader(
                    "🎯 Predicted Category"
                )

                st.markdown(
                    f"""
                    <div style="
                        padding: 25px;
                        border-radius: 12px;
                        background-color: #f0f2f6;
                        text-align: center;
                        margin-bottom: 25px;
                    ">

                    <h1>{prediction}</h1>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # TOP 3
                # ------------------------------------------------

                if top_predictions:

                    st.subheader(
                        "📊 Top 3 Possible Categories"
                    )

                    for category, probability in top_predictions:

                        percentage = (
                            probability * 100
                        )

                        col1, col2 = st.columns(
                            [3, 1]
                        )

                        with col1:

                            st.write(
                                f"**{category}**"
                            )

                            st.progress(
                                float(probability)
                            )

                        with col2:

                            st.write(
                                f"**{percentage:.2f}%**"
                            )

                # ------------------------------------------------
                # RESUME TEXT
                # ------------------------------------------------

                with st.expander(
                    "📄 View Extracted Resume Text"
                ):

                    st.text_area(
                        "Resume text",
                        resume_text,
                        height=400
                    )

                # ------------------------------------------------
                # BASIC STATISTICS
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "📈 Resume Statistics"
                )

                words = resume_text.split()

                characters = len(resume_text)

                word_count = len(words)

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Words",
                        f"{word_count:,}"
                    )

                with col2:

                    st.metric(
                        "Characters",
                        f"{characters:,}"
                    )

                with col3:

                    st.metric(
                        "Pages",
                        "PDF"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Parser • TF-IDF + Machine Learning"
)