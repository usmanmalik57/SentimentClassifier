import joblib
import os

print (os.getcwd())
tfidf_vectorizer = joblib.load(r'tfidf_vectorizer.pkl')
rf_classifier = joblib.load(r'lr_classifier.pkl')


def get_classifier_tfidf():

    return rf_classifier, tfidf_vectorizer


