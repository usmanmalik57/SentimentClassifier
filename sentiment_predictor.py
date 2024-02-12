import streamlit as st
import dataloader

# Load the classifier and TF-IDF vectorizer
rf_classifier, tfidf_vectorizer = dataloader.get_classifier_tfidf()

# Define the Streamlit app
def main():
    
    st.title("Movie Sentiment Analysis")

    user_input = st.text_area("Enter a review:")

    st.text("Message from feature 1")
    st.text("Message from feature 2")

    if st.button("Get Sentiment"):
        if user_input:
            # Predict the sentiment
            predicted_sentiment = predict_sentiment(user_input, tfidf_vectorizer, rf_classifier)

            # Display the sentiment
            st.write(f"The predicted sentiment of the review is: {predicted_sentiment}")
        else:
            st.warning("Please enter a review.")

# Predict the sentiment of a single text review
def predict_sentiment(single_review, tfidf_vectorizer, rf_classifier):
    # Preprocess the single text review (assuming you have already cleaned and preprocessed it)
    # TF-IDF Vectorization
    single_review_tfidf = tfidf_vectorizer.transform([single_review])

    # Predict the sentiment
    prediction = rf_classifier.predict(single_review_tfidf)

    return prediction[0]

if __name__ == "__main__":
    main()
