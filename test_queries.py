"""
Evaluates queries and displays retrieved documents.

Display Format:
(document id, document name, cosine similarity score)

Command Line Arguments:
    --score_title: True if zoned index considered for evaluation. Set to False by default.
    --expand_query: True if query expansion must be used. Set to False by default.
"""

import os.path
import pickle
import time
import math
import argparse
import sys
import warnings
from helper_module import processWords, printTopKScores, strToBool

def expandQuery(query_vec, kv_model, index, max_exp_no):
    '''
    Adds the synonyms of each term in query to the query vector.
    
    Parameters:
        query_vec (dictionary): The term to term frequency map of the query.
        kv_model (gensim.models.keyedvectors.KeyedVectors): The representation of each term with its corresponding vector.
        index (dictionary): The inverted index of the corpus.
        max_exp_no (int): The maximum number of synonyms to be added for each query term.
    Returns:
        query_vec (dictionary): The expanded query vector.
    '''
    temp_vec = query_vec.copy()
    
    for (term, val) in temp_vec.items():
        
        # If term is present in kv model, add the synonyms.
        try:
            # Retrieve top synonyms.
            top_nearest = kv_model.most_similar(term, topn = max_exp_no)
            
            for (near_word, near_val) in top_nearest:
                query_vec[near_word] = query_vec.get(near_word, 0) + (near_val * val)
                
        except:    
            pass
        
    for (term, val) in query_vec.items():
        
        # Limiting similarity to lie between 0 and 1
        if(val < 0):
            query_vec[term] = 0
            
    return query_vec

def constructQueryVector(query):
    '''
    Creates a mapping between each word in the query and its term frequency.
    
    Parameters:
        query (list): List of processed words from the user input query.
    Returns:
        query_vec (dictionary): The word to term frequency map of the query.
    
    '''
    query_vec = {}
    for term in query:
        query_vec[term] = query_vec.get(term, 0) + 1
    return query_vec


def computeQueryWordScore(term, tf, index, corpus_size):
    '''
    Computes the tf-idf weighting for the given term in the query.
    
    Parameters:
        term (String): The query term in consideration.
        tf (int): The term frequency obtained from the query vector.
        index (dictionary): The inverted index of the corpus.
        corpus_size (int): The number of documents in the corpus.
    Returns:
        weight (float): The tf-idf weight of the term
    
    ''' 
    df = index[term][0]
    idf = math.log((corpus_size / df), 10)
    weight = (1 + math.log(tf, 10)) * idf
    return weight


def computeDocWordScore(scores, term, qscore, index, doc_info, isTitle = False):
    '''
    Computes the score of the given term for the documents in which it occurs.
    
    Parameters:
        scores (dictionary): The current scores of each document with respect to the query.
        term (String): The query term in consideration.
        qscore (float): The tf-idf weight of the term for the query vector.
        index (dictionary): The inverted index of the corpus.
        doc_info (dictionary): The additional document info containing id, name and normalization factor.
        isTitle (Boolean): True if the scores of the documents are to be computed based on their titles.
    Returns:
        None
    
    '''
    doc_list = index[term][1]
    for (doc, val) in doc_list.items():
        tf = 1 + math.log(val, 10)
        
        # Use the document normalization factor computed during index construction
        if isTitle:
            norm_tf = tf / doc_info[doc][2]
        else:
            norm_tf = tf / doc_info[doc][1]
            
        score = qscore * norm_tf
        scores[doc] = scores.get(doc, 0) + score
            

def evaluateDocScores(query_vec, kv_model, index, doc_info, corpus_size, isTitle = False, expand_query = False):
    '''
    Constructs the dictionary of document scores.
    
    Parameters:
        query_vec (dictionary): The word to term frequency map of the query.
        index (dictionary): The inverted index of the corpus.
        doc_info (dictionary): The additional document info containing id, name and normalization factor.
        corpus_size (int): The number of documents in the corpus.
        isTitle (Boolean): True if the scores of the documents are to be computed based on their titles.
    Returns:
        scores(dictionary): The scores of each document with respect to the query.
        
    '''
    scores = {}
    query_norm = 0
    
    # Expand query if required to
    if(expand_query):
        query_vec = expandQuery(query_vec, kv_model, index, max_exp_no = 7)
    
    # Only compute score for the words in the query instead of generating the complete document vectors (index elimination)
    for word in query_vec:
        if word not in index:
            qscore = 0
        else:
            qscore = computeQueryWordScore(word, query_vec[word], index, corpus_size)
            computeDocWordScore(scores, word, qscore, index, doc_info, isTitle)
            
        # Compute the query normalization factor to be used after all documents are scored.
        query_norm += qscore ** 2
    query_norm **= 0.5
    
    # Normalize the scores using the query normalization factor
    if(query_norm > 0):
        for doc in scores:
            scores[doc] /= query_norm
    return scores


def mergeScores(doc_scores, title_scores, title_weight):
    '''
    Returns the final scores for each document after including the weighted title.
    
    Parameters:
        doc_scores (dictionary): The scores computed based on the document contents with respect to the query.
        title_scores (dictionary): The scores computed based on the document titles with respect to the query.
        title_weight (float): The weight given to the title.
    Returns:
        final_scores (dictionary): The final scores for each document after including the weighted title.
        
    '''
    final_scores =  {k: (doc_scores.get(k, 0) * (1 - title_weight)) + (title_scores.get(k, 0) * title_weight) for k in set(doc_scores) | set(title_scores)}
    return final_scores


if __name__ == "__main__":
    
    # Command Line Args for whether to include title scores or not and whether to expand query or not
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_title", type = strToBool, default = False, help = "True if title scores need to be included")
    parser.add_argument("--expand_query", type = strToBool, default = False, help = "True if query should be expanded")
    args = vars(parser.parse_args())
    if(args["score_title"] == None):
        print("Incorrect command line arguments for --score_title. Must be True/False.")
        sys.exit()
    if(args["expand_query"] == None):
        print("Incorrect command line arguments for --expand_query. Must be True/False.")
        sys.exit()
        
    # Create basic and zoned indices in different directories 
    path_to_index_file = os.path.join(os.getcwd(),'index_files')
    if(args["score_title"]):
        path_to_index = os.path.join(path_to_index_file,'zoned')
    else:
        path_to_index = os.path.join(path_to_index_file,'basic')
    
    # Load the inverted indices along with the doc_info dictionary
    try:
        with open(os.path.join(path_to_index, "inverted_index"), "rb") as file:
            inverted_index = pickle.load(file)
        with open(os.path.join(path_to_index, "doc_info"), "rb") as file:
            doc_info = pickle.load(file)
        
    except:
        print('Inverted Index not available. Please run "construct_index.py" first.')
        sys.exit()
        
    if(args["score_title"]):
        try:
            with open(os.path.join(path_to_index, "title_index"), "rb") as file:
                title_index = pickle.load(file)
        except:
            print('Zoned Inverted Index not available. Please run "construct_index.py --zoned_index True" first.')
            sys.exit()
    
    kv_model = None
    if(args["expand_query"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(os.path.join(path_to_index_file, "kv_model"), "rb") as file:
                    kv_model = pickle.load(file)
        except:
            print('Word Embeddings not available. Please run "trim_embeddings.py" first.')
            sys.exit()
        
    corpus_size = len(doc_info)
        
    while True:
        query = input("Enter query here: ")
        
        start = time.time()
        
        # Clean the query words and construct the query vector
        query = processWords(query)
        query_vec = constructQueryVector(query)
        
        # Score the docs based on content and title (if required)
        doc_scores = evaluateDocScores(query_vec, kv_model, inverted_index, doc_info, corpus_size, expand_query = args["expand_query"])
        if(args["score_title"]):
            title_scores = evaluateDocScores(query_vec, kv_model, title_index, doc_info, corpus_size, isTitle = True, expand_query = False)
            doc_scores = mergeScores(doc_scores, title_scores, title_weight = 0.1)
        
        end = time.time()
        
        # Print top K docs in the required format
        printTopKScores(doc_info, doc_scores, K = 10, total_time = end - start)
        
        print("")
        if(input("Do you want to continue? Enter y/n: ") == "n"):
            break
        print("\n\n")