# sms_spam_detector.py

import pandas as pd
import streamlit as st
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Map labels to 0 and 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label_num'], test_size=0.2, random_state=42)

# Create TF-IDF + Naive Bayes pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Save model
with open('spam_detector.pkl', 'wb') as f:
    pickle.dump(model, f)

# Check accuracy
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Streamlit App
st.title("📱 SMS Spam Detector")
input_sms = st.text_area("Enter your message")

if st.button("Check"):
    result = model.predict([input_sms])[0]
    st.subheader("Result:")
    st.write("🔴 Spam" if result == 1 else "🟢 Not Spam")
