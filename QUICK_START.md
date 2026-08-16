# Quick Start Guide - IMDB Sentiment Analyzer

## 🚀 Quick Setup (5 minutes)

### 1. Navigate to Project Directory
```powershell
cd "c:\Users\Jevil\OneDrive\Desktop\ML\New folder\deep learning\rnn"
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the App
```powershell
streamlit run app.py
```

### 4. Open Browser
The app opens automatically at: `http://localhost:8501`

---

## 📝 Deployment Steps Summary

| Step | Action | Command |
|------|--------|---------|
| 1 | Save model & vectorizer | ✅ Already done in notebook |
| 2 | Install packages | `pip install -r requirements.txt` |
| 3 | Run locally | `streamlit run app.py` |
| 4 | Test the app | Enter a review and click "Analyze Sentiment" |
| 5 | Deploy online | See DEPLOYMENT_GUIDE.md |

---

## 📂 Project Files

```
rnn/
├── app.py                          # Streamlit application
├── rnn.ipynb                       # Training notebook
├── rnn_model.pth                   # Trained model weights
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── IMDB Dataset.csv                # Training data
├── requirements.txt                # Python dependencies
├── DEPLOYMENT_GUIDE.md             # Full deployment guide
└── QUICK_START.md                  # This file
```

---

## 🎯 Model Features

- **Input**: Movie review text (any length)
- **Output**: Sentiment prediction (Positive/Negative with confidence)
- **Preprocessing**: Automatic text cleaning and preprocessing
- **Model Type**: RNN (Recurrent Neural Network) with PyTorch

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError" | Run `pip install -r requirements.txt` |
| "FileNotFoundError: rnn_model.pth" | Make sure all files are in same directory |
| "FileNotFoundError: tfidf_vectorizer.pkl" | Run the last cell of rnn.ipynb |
| App slow on first load | Normal - model is cached after first load |

---

## 📚 File Descriptions

### app.py
Streamlit web application with:
- Text input for reviews
- Model prediction logic
- Preprocessing pipeline
- Results visualization

### rnn.ipynb
Jupyter notebook containing:
- Data loading and exploration
- Text preprocessing
- Model training
- Model evaluation
- Export functionality

### rnn_model.pth
Binary file containing:
- Trained RNN model weights
- Model parameters
- Ready for inference

### tfidf_vectorizer.pkl
Pickle file containing:
- Fitted TF-IDF vectorizer
- Vocabulary (5000 features)
- Used for text transformation

---

## 🌐 Deploy Online

Choose one option:

**Option 1: Streamlit Cloud** (Easiest)
- Upload to GitHub
- Connect at https://streamlit.io/cloud
- Auto-deploys on push

**Option 2: Heroku**
- Create `Procfile`
- Deploy via `git push heroku main`

**Option 3: AWS/Azure/Google Cloud**
- Create Docker container
- Deploy to your cloud provider

See DEPLOYMENT_GUIDE.md for detailed instructions.

---

## 💡 Tips

- Test locally first before deploying
- Keep all files in the same directory
- Model loads only once (cached for performance)
- Preprocessing is identical to training pipeline
- App handles long reviews (splits into chunks if needed)

---

**Ready to go? Run `streamlit run app.py` now!** 🚀
