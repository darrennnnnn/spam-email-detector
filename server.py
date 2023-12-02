# backend.py
from flask import Flask, request, jsonify
import pickle  # Use pickle to load the .pkl file
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)


# Importing the Porter Stemmer for text stemming
from nltk.stem.porter import PorterStemmer

# Importing the string module for handling special characters
import string

# Creating an instance of the Porter Stemmer
ps = PorterStemmer()

# Lowercase transformation and text preprocessing function
def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()
    
    # Tokenization using NLTK
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing stop words and punctuation
    text = y[:]
    y.clear()
    
    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    # Join the processed tokens back into a single string
    return " ".join(y)

# Load your pre-trained machine learning model
with open('../svcmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('../vectorizer.pkl', 'rb') as model_file:
    tfid = pickle.load(model_file)

@app.route('/wow', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)
    text = data.get('text', '')
    
    if text:
        transformed_data = transform_text(text)

        vector_input = tfid.transform([transformed_data])

        result = model.predict(vector_input)[0]

        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
