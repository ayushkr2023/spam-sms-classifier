# SMS/Email Spam Classifier

A Streamlit-based web app that classifies SMS and Email messages as Spam or Not Spam using Natural Language Processing (NLP) and a trained Multinomial Naive Bayes model.

---

## Demo
After deployment, your app will be live at:  
https://<your-app-name>.streamlit.app

---

## Features
- Classifies text messages (SMS/Email) as Spam or Not Spam  
- Text preprocessing with:
  - Lowercasing  
  - Tokenization  
  - Stopword removal  
  - Punctuation removal  
  - Stemming (Porter Stemmer)  
- TF-IDF vectorization for feature extraction  
- Trained Naive Bayes model from scikit-learn  
- Simple and interactive Streamlit UI

---

## Tech Stack
- Python 3.11  
- Streamlit  
- Scikit-learn  
- NLTK  
- Pickle (for saving model/vectorizer)

---

## Project Structure

spam-classifier/
│── app.py              # Streamlit app
│── vectorizer.pkl      # Saved TF-IDF vectorizer
│── model.pkl           # Trained Naive Bayes model
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation


## Sample Inputs/Outputs
	•	“Congratulations! You’ve won a free lottery ticket. Click here to claim.” → Spam
	•	“Are we still meeting at 5pm today?” → Not Spam
	•	“Limited time offer!! Buy 1 Get 1 Free!!” → Spam
	•	“Please send me the report by tomorrow morning.” → Not Spam
 
---
## Installation & Usage

### 1.Clone the repository
 ```bash
git clone https://github.com/<your-username>/sms_spam_detector.git
cd sms_spam_detector

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

streamlit run app.py

App will be available at:
http://localhost:8501/
