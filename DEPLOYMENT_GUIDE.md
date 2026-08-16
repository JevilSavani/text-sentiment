# IMDB Sentiment Analysis - Streamlit Deployment Guide

## Project Overview
This Streamlit application deploys an RNN model trained on IMDB movie reviews to predict sentiment (Positive/Negative) from user input comments.

---

## Files Included
- `rnn.ipynb` - Jupyter notebook with model training
- `app.py` - Streamlit application
- `rnn_model.pth` - Saved trained model weights
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer
- `requirements.txt` - Python dependencies
- `IMDB Dataset.csv` - Training dataset
- `DEPLOYMENT_GUIDE.md` - This file

---

## Step 1: Save the TF-IDF Vectorizer (Important!)

Before deploying, you need to save the TF-IDF vectorizer used during training. Add this code to your Jupyter notebook:

```python
import pickle

# After training (in your notebook)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf, f)
```

**Action:** Run this cell in your notebook to create `tfidf_vectorizer.pkl`

---

## Step 2: Verify Required Files

Make sure these files exist in your project directory:
- ✅ `rnn_model.pth` (model weights) - Already exported
- ✅ `tfidf_vectorizer.pkl` (TF-IDF vectorizer) - Need to create (Step 1)
- ✅ `app.py` (Streamlit app) - Already created
- ✅ `requirements.txt` (dependencies) - Already created

---

## Step 3: Install Dependencies

Open PowerShell/Command Prompt and navigate to your project directory:

```powershell
cd "c:\Users\Jevil\OneDrive\Desktop\ML\New folder\deep learning\rnn"
```

Install all required packages:

```powershell
pip install -r requirements.txt
```

---

## Step 4: Run the Streamlit Application

```powershell
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## Step 5: Using the Application

1. **Enter a Review**: Type or paste a movie review in the text area
2. **Click "Analyze Sentiment"**: Submit for prediction
3. **View Results**:
   - **Positive Score**: Probability of positive sentiment (0-100%)
   - **Sentiment**: Overall classification (Positive/Negative)
   - **Confidence**: How confident the model is
   - **Analysis Details**: Original vs. preprocessed text

---

## Deployment to Production

### Option A: Deploy on Streamlit Cloud (Recommended for Beginners)

1. **Upload to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/imdb-sentiment.git
   git push -u origin main
   ```

2. **Create Streamlit Account**:
   - Go to https://streamlit.io/cloud
   - Sign up with GitHub account

3. **Deploy**:
   - Click "New app"
   - Select your GitHub repository
   - Select the branch and file (`app.py`)
   - Click "Deploy"

### Option B: Deploy on Heroku

1. **Create `Procfile`**:
   ```
   web: streamlit run --logger.level=error --server.port=$PORT app.py
   ```

2. **Create `.gitignore`**:
   ```
   __pycache__/
   *.pyc
   .streamlit/
   ```

3. **Deploy**:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Option C: Deploy on AWS/Azure/Google Cloud

1. Create a Docker container with the app
2. Deploy using respective cloud services

---

## Troubleshooting

### Error: "FileNotFoundError: rnn_model.pth not found"
- **Solution**: Make sure `rnn_model.pth` is in the same directory as `app.py`

### Error: "FileNotFoundError: tfidf_vectorizer.pkl not found"
- **Solution**: Run the code in Step 1 to save the vectorizer from your notebook

### Error: "NLTK data not found"
- **Solution**: The app automatically downloads NLTK data on first run
- If it fails, manually download:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

### App runs slow on first load
- **Normal behavior**: The model is cached after first load for better performance

---

## Model Architecture

```
Input (5000 features from TF-IDF)
    ↓
RNN Layer (hidden_size=128, num_layers=1)
    ↓
Fully Connected Layer (128 → 1)
    ↓
Sigmoid Activation
    ↓
Output (0-1 probability)
```

---

## Text Preprocessing Pipeline

The app applies the same preprocessing as training:

1. **Lowercase**: Convert text to lowercase
2. **Remove URLs**: Strip HTTP/HTTPS links
3. **Remove Punctuation**: Keep only alphanumeric characters
4. **Remove HTML**: Strip HTML tags
5. **Remove Stopwords**: Remove common English words
6. **Stemming**: Reduce words to root form (e.g., "running" → "run")

---

## Performance Metrics (From Training)

- **Model Accuracy**: Check your notebook for training results
- **Test Set Performance**: ~[Your Accuracy]%

---

## API Usage (Optional: Extend App)

You can extend the app to include an API endpoint:

```python
from streamlit_server_state import server_state

# Add FastAPI wrapper for REST endpoint access
```

---

## Maintenance & Updates

### To retrain the model:
1. Modify `rnn.ipynb` with new hyperparameters or data
2. Re-run all cells
3. Re-export:
   ```python
   torch.save(model.state_dict(), 'rnn_model.pth')
   pickle.dump(tf, open('tfidf_vectorizer.pkl', 'wb'))
   ```
4. Redeploy the app

---

## Next Steps

- ✅ Save TF-IDF vectorizer (Step 1)
- ✅ Test locally with `streamlit run app.py`
- ✅ Deploy to Streamlit Cloud or Heroku
- ✅ Share the link with others!

---

## Contact & Support

For issues or questions:
- Check the app logs: `streamlit run app.py --logger.level=debug`
- Review Streamlit docs: https://docs.streamlit.io

---

**Happy Deploying! 🚀**
