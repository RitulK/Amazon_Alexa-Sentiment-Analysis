# import joblib

# try:
#     # Use joblib to load the model
#     model = joblib.load("Models/random_forest_model_for_download.joblib")
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Load the models
predictor = pickle.load(open(r"Models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

# Sample review
sample_review = "The product is amazing, I love it!"

# Preprocessing
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
review = re.sub("[^a-zA-Z]", " ", sample_review)
review = review.lower().split()
review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
review = " ".join(review)
print(f"Preprocessed Review: {review}")

# Transform the review into features
X_prediction = cv.transform([review]).toarray()
X_prediction_scl = scaler.transform(X_prediction)

# Predict sentiment
y_predictions = predictor.predict_proba(X_prediction_scl)
print(f"Prediction Probabilities: {y_predictions}")
y_predictions = y_predictions.argmax(axis=1)[0]

print("Sentiment:", "Positive" if y_predictions == 1 else "Negative")
