import pandas as pd
import re

# Load the cleaned resume dataset
df = pd.read_csv("../dataset/Resume_clean.csv")

# Function to clean resume text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # Keep letters and spaces
    text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces
    return text

# Apply cleaning
df["clean_resume"] = df["Resume_str"].apply(clean_text)

# Save the processed dataset
df.to_csv("../dataset/Resume_processed.csv", index=False)

print("Preprocessing completed!")
print("Total resumes:", len(df))
print(df[["Category", "clean_resume"]].head())