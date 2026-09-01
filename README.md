# AI Resume Parser

An AI-powered Resume Parser that uses Natural Language Processing, TF-IDF, and Machine Learning to analyze PDF resumes and predict the most likely professional category.

## Project Overview

The application allows users to upload a PDF resume. It then:

1. Extracts text from the PDF.
2. Cleans and preprocesses the text.
3. Converts the text into numerical features using TF-IDF.
4. Uses a trained Machine Learning classification model.
5. Predicts the most likely professional category.
6. Displays the Top 3 predicted categories with confidence scores.
7. Shows basic resume statistics such as word count and character count.

## Project Architecture

```text
Resume_Parser_AI/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── resume_category_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── dataset/
│   ├── Resume.csv
│   ├── Resume_clean.csv
│   ├── Resume_processed.csv
│   └── Resume_final.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Workflow

```text
Resume Dataset
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
      ↓
Save Model and Vectorizer
      ↓
Streamlit Application
      ↓
Upload PDF Resume
      ↓
Extract and Preprocess Text
      ↓
TF-IDF Transformation
      ↓
Category Prediction
```

## Dataset

The project uses a resume dataset containing resume text and professional categories.

The dataset is processed through multiple stages:

* `Resume.csv` - Original dataset
* `Resume_clean.csv` - Cleaned dataset
* `Resume_processed.csv` - Preprocessed dataset
* `Resume_final.csv` - Final dataset used for Machine Learning

## Text Preprocessing

Resume text is cleaned before being used by the Machine Learning model.

The preprocessing includes:

* Converting text to lowercase
* Removing unnecessary characters
* Removing extra spaces
* Removing text noise
* Preparing text for TF-IDF vectorization

The same preprocessing approach is applied to new resumes during prediction.

## TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts resume text into numerical features.

It gives greater importance to words that help distinguish between different professional categories.

The trained vectorizer is saved as:

```text
app/tfidf_vectorizer.pkl
```

## Machine Learning Model

A trained Machine Learning classification model is used to predict the professional category of a resume.

The trained model is saved as:

```text
app/resume_category_model.pkl
```

The prediction process is:

```text
Resume Text
     ↓
Preprocessing
     ↓
TF-IDF Vectorizer
     ↓
Machine Learning Model
     ↓
Predicted Category
```

## Streamlit Application

The user interface is built using Streamlit.

The application provides:

* PDF resume upload
* Resume text extraction
* Professional category prediction
* Top 3 predictions with confidence scores
* Resume word count
* Resume character count

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* PyPDF2
* Streamlit
* TF-IDF
* Machine Learning

## Installation

Clone the repository:

```bash
git clone https://github.com/ameenafirdous/Resume_Parser_AI.git
```

Move into the project directory:

```bash
cd Resume_Parser_AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will usually be available at:

```text
http://localhost:8501
```

## Streamlit Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment configuration:

```text
Repository: ameenafirdous/Resume_Parser_AI
Branch: main
Main file: app/app.py
```

The `requirements.txt` file contains the Python packages required to run the application.

## Example Output

After uploading a resume, the application may produce an output such as:

```text
Predicted Category

BANKING

Top 3 Possible Categories

BANKING                 36.12%
BUSINESS-DEVELOPMENT     9.21%
CONSULTANT               5.61%
```

The actual predictions and confidence scores depend on the uploaded resume and trained model.

## Project Objective

The objective of this project is to demonstrate how Natural Language Processing and Machine Learning can be used to classify resumes into professional categories.

The project combines PDF processing, text preprocessing, TF-IDF, Machine Learning, and Streamlit into an end-to-end AI Resume Parser.

## Future Improvements

* Improve classification accuracy
* Support additional resume formats
* Add skill extraction
* Extract education and work experience
* Extract contact information
* Add job-role recommendations
* Add resume scoring
* Compare multiple Machine Learning models
* Use a larger and more balanced dataset

## Author

Ameena Firdous

GitHub: https://github.com/ameenafirdous/Resume_Parser_AI

## License

This project is intended for educational and demonstration purposes.
