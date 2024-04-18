from flask import Flask, request, jsonify
from summa.summarizer import summarize
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

@app.route('/')
def index():
    return 'Welcome to the Semantic Text Similarity API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read JSON request data
        data = request.json
        text1 = data['text1']
        text2 = data['text2']

        # Summarize text
        text1 = summarize_text(text1)
        text2 = summarize_text(text2)

        # Preprocess text
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

        # Encode texts into BERT embeddings
        embeddings1 = model.encode(text1)
        embeddings2 = model.encode(text2)

        # Calculate cosine similarity between embeddings
        similarity_score = cosine_similarity([embeddings1], [embeddings2])[0][0]

        # Convert similarity score to float
        similarity_score = float(similarity_score), 

        # Return JSON response
        response = {'similarity_score': similarity_score}
        return jsonify(response)
    
    except Exception as e:
        # Error handling: Print and return error message
        error_message = str(e)
        print("Error:", error_message)
        return jsonify({'error': error_message})

def summarize_text(text):
    summary = summarize(text, ratio=0.2)
    return summary if summary else text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    tokens = [token for token in tokens if token not in string.punctuation]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


if __name__ == '__main__':
    app.run(debug=True)
