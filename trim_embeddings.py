"""
Trims the GloVe embeddings to contain terms only from corpus.
"""

import os
import sys
import os.path
import pickle
import numpy as np
import warnings
from tqdm import tqdm
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gensim
    from gensim.models.keyedvectors import KeyedVectors


if __name__ == "__main__":
    
    # Load the raw embeddings
    print("Loading dataset...")
    try:
        raw_w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary = False, no_header = True)
    except:
        print('GloVe embeddings not available. Please download them as given in the readme')
        sys.exit()
    
    path_to_index_files = os.path.join(os.getcwd(),'index_files')
    
    # Load inverted index
    try:
        with open(os.path.join(path_to_index_files, "basic", "inverted_index"), "rb") as file:
            inverted_index = pickle.load(file)
    except:
        print('Inverted Index not available. Please run "construct_index.py" first.')
        sys.exit()
        
    vector_length = raw_w2v.vector_size
    kv_model = KeyedVectors(vector_length)
    vector_list = []
    word_list = []
    
    for word in tqdm(inverted_index, desc = "Computing word similarities", ascii = False, ncols = 100, file = sys.stdout):
        
        # Only consider the terms present in the corpus vocabulary 
        try:
            vector = raw_w2v.get_vector(word)
            word_list.append(word)
            vector_list.append(vector)
        except:
            pass

    kv_model.add_vectors(word_list, vector_list)

    # Store the trimmed model
    with open(os.path.join(path_to_index_files, "kv_model"), "wb") as file:
        pickle.dump(kv_model, file)
    
    print("\nEmbeddings extracted and saved to disk")