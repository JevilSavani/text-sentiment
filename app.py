import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Text Preprocessing Functions
def remove_url(text):
    return re.sub(r"http\S+", "", text)


def remove_punctuation(text):
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_stopwords_func(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    
    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")
    return text


def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []
    tokens = word_tokenize(text)
    
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)
    
    return " ".join(stemmed_words)


def preprocess_text(text):
    """Complete preprocessing pipeline"""
    text = text.lower()
    text = remove_url(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_stopwords_func(text)
    text = stemming(text)
    return text


@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        # Load TF-IDF Vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Model
        model = RNN(input_size=5000)
        model.load_state_dict(torch.load('rnn_model.pth'))
        model.eval()
        
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None


def predict_sentiment(comment, model, vectorizer):
    """Predict sentiment for a given comment"""
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded!")
        return None
    
    # Preprocess the comment
    preprocessed_comment = preprocess_text(comment)
    
    # Transform using TF-IDF
    comment_vector = vectorizer.transform([preprocessed_comment])
    comment_array = comment_vector.toarray()
    
    # Convert to torch tensor
    comment_tensor = torch.tensor(comment_array, dtype=torch.float32)
    comment_tensor = comment_tensor.unsqueeze(1)
    
    # Make prediction
    with torch.no_grad():
        output = model(comment_tensor)
        prediction = torch.sigmoid(output.squeeze()).item()
    
    return prediction


# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="wide")

st.title("🎬 IMDB Sentiment Analysis")
st.markdown("---")
st.write("**Analyze movie reviews using a trained RNN model**")
st.markdown("---")

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

if model is not None and vectorizer is not None:
    # Create input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        comment = st.text_area(
            "Enter your movie review:",
            height=150,
            placeholder="Type your review here..."
        )
    
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("Analyze Sentiment", use_container_width=True)
    
    st.markdown("---")
    
    # Make prediction
    if predict_button and comment.strip():
        prediction = predict_sentiment(comment, model, vectorizer)
        
        if prediction is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Positive Score", f"{prediction:.2%}")
            
            with col2:
                sentiment = "Positive 👍" if prediction > 0.5 else "Negative 👎"
                st.metric("Sentiment", sentiment)
            
            with col3:
                confidence = abs(prediction - 0.5) * 2 * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            st.markdown("---")
            
            # Show detailed analysis
            st.subheader("Analysis Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Review:**")
                st.info(comment)
            
            with col2:
                st.write("**Preprocessed Review:**")
                preprocessed = preprocess_text(comment)
                st.info(preprocessed)
            
            # Show prediction details
            st.write("**Prediction Details:**")
            st.write(f"- Positive Probability: **{prediction:.4f}**")
            st.write(f"- Negative Probability: **{1-prediction:.4f}**")
            
            if prediction > 0.7:
                st.success("✅ Strong Positive Sentiment")
            elif prediction > 0.5:
                st.success("✔️ Positive Sentiment")
            elif prediction > 0.3:
                st.warning("⚠️ Negative Sentiment")
            else:
                st.error("❌ Strong Negative Sentiment")
    
    elif predict_button and not comment.strip():
        st.warning("Please enter a review to analyze!")
else:
    st.error("⚠️ Could not load model or vectorizer. Please ensure 'rnn_model.pth' and 'tfidf_vectorizer.pkl' are in the same directory as this app.")
