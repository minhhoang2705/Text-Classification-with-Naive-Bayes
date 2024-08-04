import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('SPAM text message 20170820 - Data.csv')
    return df

df = load_data()

# Preprocess data
df['Message'] = df['Message'].str.replace("[^a-zA-Z0-9 ]", "")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

# Streamlit app
def main():
    st.title('SMS Spam Classifier')
    new_message = st.text_area("Enter a SMS message to classify:")
    if st.button("Classify"):
        # Preprocess the input text
        processed_input = new_message.replace("[^a-zA-Z0-9 ]", "")
        input_vec = vectorizer.transform([processed_input])
        
        # Make prediction
        prediction = model.predict(input_vec)
        if prediction[0] == 'spam':
            st.write('This message is classified as SPAM.')
        else:
            st.write('This message is classified as HAM (non-spam).')

if __name__ == "__main__":
    main()
