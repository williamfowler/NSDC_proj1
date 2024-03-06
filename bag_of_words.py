from sklearn.feature_extraction.text import CountVectorizer
import pickle


def feature_extractor1(x_train_df): 
    with open('vocab.pkl','rb') as f:
        vocabulary = pickle.load(f)
    x_text = []
    for n in x_train_df:
        n.pop(0)
        x_text.append(n[0])

    vectorizer = CountVectorizer(vocabulary=vocabulary)

    x = vectorizer.transform(x_text)  
    return x