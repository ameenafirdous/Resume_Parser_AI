import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the cleaned dataset
df = pd.read_csv(r"..\dataset\Resume_final.csv")

# 2. Get resume text and category
X = df["clean_resume"].astype(str)
y = df["Category"].astype(str)


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training resumes:", len(X_train))
print("Testing resumes:", len(X_test))


# 4. Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Train the AI model
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# 6. Test the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7. Save the trained model
joblib.dump(model, "resume_category_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("resume_category_model.pkl")
print("tfidf_vectorizer.pkl")