import joblib
import os

print (os.getcwd())
tfidf_vectorizer = joblib.load(r'lr_classifier.pkl')
rf_classifier = joblib.load(r'lr_classifier.pkl')

# Predict the sentiment of a single text review using the trained classifier
def predict_sentiment(single_review, tfidf_vectorizer, rf_classifier):
    # Preprocess the single text review (assuming you have already cleaned and preprocessed it)
    # TF-IDF Vectorization
    single_review_tfidf = tfidf_vectorizer.transform([single_review])

    # Predict the sentiment
    prediction = rf_classifier.predict(single_review_tfidf)


    return prediction[0]

# Predict the sentiment of a single text review in production
single_review = "Your single text review goes here"
predicted_sentiment = predict_sentiment(single_review, tfidf_vectorizer, rf_classifier)

print(f"The predicted sentiment of the review is: {predicted_sentiment}")