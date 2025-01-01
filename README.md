****Sentiment Prediction Portal for Amazon Alexa Reviews****

Project Overview:

This project performs sentiment analysis on Amazon Alexa reviews to classify feedback as positive or negative. It utilizes a pre-trained machine learning model (XGBoost) and provides an interactive web portal built using Flask, where users can input text reviews to determine their sentiment.

Features:

Sentiment Analysis: Predicts whether a given text review is positive or negative.

Interactive Portal: A user-friendly web interface where users can input text and view predictions.

Pre-trained Model: Uses XGBoost for high-performance sentiment prediction.

Project Components:

Model Training (IPython Notebook):

Objective: Train machine learning models (Random Forest and XGBoost) to classify reviews.

Dataset: Amazon Alexa user reviews dataset.

Attributes include Rating, Reviews, Date, Variation, and Feedback.

Sentiment labels were created from the Rating column (4-5: Positive, 1-3: Negative).

Algorithms Used:

Random Forest: Training Accuracy (99.45%), Testing Accuracy (93.96%).

XGBoost: Training Accuracy (97.14%), Testing Accuracy (94.17%).

Artifacts: The trained XGBoost model, CountVectorizer, and Scaler were saved as .pkl files for deployment.

Flask Application:

api.py:

Provides a REST API endpoint for sentiment prediction.

Loads the pre-trained XGBoost model along with CountVectorizer and Scaler.

Preprocesses input text and returns the sentiment prediction.

main.py:

Implements the front-end portal using Flask.

Displays a form where users can input text.

Shows the sentiment prediction for the entered review.

Templates:

HTML templates for the landing page and results.

Dataset Details

Source: Amazon Alexa user reviews dataset.

Attributes:

Rating: Review rating on a scale from 1 to 5.

Reviews: Text reviews given by the customers.

Date: When the review was posted.

Variation: Different product variations reviewed.

Target: Sentiment label (positive or negative) created from the Rating column.

Reviews with ratings 4 and 5 were labeled positive.

Reviews with ratings 1 to 3 were labeled negative.

Technologies Used

Machine Learning: XGBoost for model training.

Preprocessing: CountVectorizer for text vectorization, Scaler for feature scaling.

Web Framework: Flask for building the interactive portal.

![Screenshot 2024-12-31 175648](https://github.com/user-attachments/assets/25afada3-e7cc-4fd3-8efe-e261882533df)

