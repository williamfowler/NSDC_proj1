import pickle
import numpy as np
from gensim.models import Word2Vec

# takes average of vector for all words in a sentence
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector

def feature_extractor2(x_train_df):
    with open('word2vec_model.pkl','rb') as f:
        word_2_vec = pickle.load(f)
    # print(x_train_df)
    x_text = []
    for n in x_train_df:
        # n.pop(0)
        x_text.append(n[1])
    
    feature_vectors = [average_word_vectors(review, word_2_vec, word_2_vec.wv.key_to_index, 100) for review in x_text]
    return np.stack(feature_vectors)


