# Tweet Like Predictor

CAIC Summer of Tech 2025 – Week 2 Submission  
Track: Machine Learning + Development  
Project: Predicting Tweet Likes using Metadata

---

## Project Overview

The objective of this project is to train a machine learning regression model that predicts the number of likes a tweet will get based on available metadata. The trained model is deployed as a Flask API with a basic HTML frontend for user input.

---

## Features Used in the Final Model

After experimenting with various combinations of features, we finalized the following:

- `word_count`: Number of words in the tweet  
- `char_count`: Number of characters in the tweet  
- `has_media`: Binary indicator for media (0 = No, 1 = Yes)  
- `hour`: Hour of the day (0–23)  
- `sentiment`: Sentiment polarity score (0 to 1)  
- `company_encoded`: Encoded value of the company mentioned  
- `username_encoded`: Encoded value of the tweeting account  
- `day_of_week`: Day of the week (0 = Monday, ..., 6 = Sunday)

These were selected for their availability **prior to tweet publication** and relative importance in early model experiments.

---

## Why TF-IDF Was Not Used in the Final Model

TF-IDF (Term Frequency-Inverse Document Frequency) was initially tested as a textual feature. However:

1. **Large Feature Space**: TF-IDF creates thousands of features for each unique word/phrase in the text, leading to a very high-dimensional sparse matrix.  
2. **Exaggerated Influence**: Since tweet content was short, a few rare terms dominated the TF-IDF score, leading to disproportionately large outputs and unstable predictions (sometimes in thousands of likes).
3. **Incompatibility with Metadata-only Inputs**: The final API was designed to work **only with metadata**, and not require the actual tweet text. TF-IDF depends on the text, so we excluded it to keep things consistent and lightweight.

---

## Model Training Summary

- **Model Used**: `XGBoostRegressor`
- **Target Variable**: `log(1 + likes)` to normalize skewed distribution
- **Data Split**: 80% training / 20% testing
- **Performance**: RMSE ≈ 1.28 (log scale), performs well on most metadata

---

## Sample Prediction (Interpretation)

**Input:**

```json
{
  "word_count": 28,
  "char_count": 420,
  "has_media": 1,
  "hour": 21,
  "sentiment": 0.95,
  "company_encoded": 8,
  "username_encoded": 11,
  "day_of_week": 5
}
```
**Output:**
```json
Predicted Likes: 395
```

Interpretation:
This tweet is expected to do well — it was posted late in the evening (hour 21), has high sentiment, and contains media. The model correctly interprets it as a potentially viral tweet.

## How the API Works

1)User enters tweet metadata in the HTML form.

2)The data is sent as a POST request to the /predict endpoint.

3)The Flask server loads the trained model and reshapes the input.

4)The model predicts log(1 + likes) → converted back to actual likes using expm1.

5)The result is displayed on the page as predicted likes.

## Running the API Locally
*Requirements:*

Python 3.x

Flask, XGBoost, NumPy, scikit-learn, joblib

**Steps:**
```json
pip install flask xgboost numpy scikit-learn joblib

cd path/to/flask_project
python like_predictor_api.py
```
Visit http://127.0.0.1:5000 in your browser to test it.

## Notes and Future Improvements

-Incorporating TF-IDF in a separate pipeline with text pre-processing and dimensionality reduction (e.g., PCA) might stabilize its impact.

-Adding user-level statistics (follower count, historical engagement) would improve prediction quality.

-Incorporating tweet text via pre-trained models like BERT or DistilBERT could lead to major gains in accuracy.

-More advanced encoding strategies (like frequency or target encoding) for company and username can be explored.

## Deliverables

-Trained model: like_predictor.pkl

-Working Flask API: like_predictor_api.py + index.html

-This README with feature documentation and improvement notes

