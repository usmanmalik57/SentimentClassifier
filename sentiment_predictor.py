import dataloader


rf_classifier, tfidf_vectorizer = dataloader.get_classifier_tfidf()



# Predict the sentiment of a single text review using the trained classifier
def predict_sentiment(single_review, tfidf_vectorizer, rf_classifier):
    # Preprocess the single text review (assuming you have already cleaned and preprocessed it)
    # TF-IDF Vectorization
    single_review_tfidf = tfidf_vectorizer.transform([single_review])

    # Predict the sentiment
    prediction = rf_classifier.predict(single_review_tfidf)


    return prediction[0]

# Predict the sentiment of a single text review in production
single_review = "The cell phone is too bad"
predicted_sentiment = predict_sentiment(single_review, tfidf_vectorizer, rf_classifier)

print(f"The predicted sentiment of the review is: {predicted_sentiment}")