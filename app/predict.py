import joblib
from pypdf import PdfReader
import re

# Load model and vectorizer
model = joblib.load("resume_category_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def clean_text(text):
    """Clean resume text in the same way as training data."""

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("=" * 55)
print("           AI RESUME CATEGORY PREDICTOR")
print("=" * 55)

pdf_path = input("\nEnter the FULL path of your PDF resume:\n\n")

try:

    # Extract PDF text
    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text.strip():
        print("\nERROR: Could not extract text from this PDF.")
        print("The PDF may be scanned/image-based.")
        exit()

    print("\nResume text extracted successfully!")

    # Clean text
    cleaned_resume = clean_text(resume_text)

    # Convert text to TF-IDF
    resume_vector = vectorizer.transform([cleaned_resume])

    # Predict
    prediction = model.predict(resume_vector)[0]

    # Probabilities
    probabilities = model.predict_proba(resume_vector)[0]

    # Top 3 predictions
    top_indices = probabilities.argsort()[-3:][::-1]

    print("\n" + "=" * 55)
    print("                 RESULT")
    print("=" * 55)

    print(f"\nPREDICTED CATEGORY: {prediction}")

    print("\nTOP 3 POSSIBLE CATEGORIES:")

    for index in top_indices:
        category = model.classes_[index]
        confidence = probabilities[index] * 100

        print(f"{category:<25} {confidence:.2f}%")

    print("\n" + "=" * 55)

except FileNotFoundError:

    print("\nERROR: PDF file was not found.")
    print("Check the file path and try again.")

except Exception as e:

    print("\nERROR:", e)