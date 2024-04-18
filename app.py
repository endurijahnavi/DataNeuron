from flask import Flask, request, jsonify, render_template
from summa.summarizer import summarize
import torch
from transformers import BertTokenizer, BertModel
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

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity_score = None
    
    if request.method == 'POST':
        try:
            # Read form data
            text1 = request.form['text1']
            text2 = request.form['text2']

            # Summarize text
            text1 = summarize_text(text1)
            text2 = summarize_text(text2)

            # Preprocess text
            text1 = preprocess_text(text1)
            text2 = preprocess_text(text2)

            # Encode texts into BERT embeddings
            embeddings1 = get_bert_embeddings(text1)
            embeddings2 = get_bert_embeddings(text2)

            # Calculate cosine similarity between embeddings
            similarity_score = cosine_similarity([embeddings1], [embeddings2])[0][0]

            # Convert similarity score to float
            similarity_score = float(similarity_score)

        except Exception as e:
            # Error handling: Print and return error message
            error_message = str(e)
            print("Error:", error_message)

    return render_template('index.html', similarity_score=similarity_score)

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

def get_bert_embeddings(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling

    return embeddings

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port= 8080)
