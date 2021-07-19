"""
Contains functions used by other files.

This is a helper module only.
Do not run this file.
"""

import string
import heapq
import re
import distutils
from operator import itemgetter

def strToBool(s):
    '''
    Converts a variable to True/False boolean values. Returns None if not possible.
    
    Parameters:
        s (String/bool): The variable to be converted to boolean.
    Returns:
        True/False if conversion successful. None if not successfull.
        
    '''
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        return None


def createChampionList(posting_list, max_docs):
    '''
    Modifies the given posting list of a particular term into a champion list.
    
    Parameters:
        posting_list (dictionary): Maps document ID to term frequency.
        max_docs (int): The maximum number of documents required in the champion list.
    Returns:
        champion_list (dictionary): Reverse sorted and trimmed posting list
        
    '''
    #Extract top n docs without sorting
    sorted_tup = heapq.nlargest(max_docs, posting_list.items(), key = itemgetter(1))
    
    champion_list = {}
    for (docId, tf) in sorted_tup:
        champion_list[docId] = tf
    return champion_list


def printTopKScores(doc_info, scores, K, total_time):
    '''
    Takes the dictionary of scores and prints the docs with top K scores.
    
    Parameters:
        doc_info (dictionary): The additional document info containing id, name and normalization factor.
        scores (dictionary): The scores of each document with respect to the query.
        K (int): The number of docs to be printed.
        total_time (int): The time taken for retrieving documents.
    Returns:
        None
    
    '''
    #Extract top k docs without sorting
    top_k_docs = heapq.nlargest(K, scores, key = scores.get)
    
    #Print the top k document ids, names and scores.
    if(len(top_k_docs) > 0):
        print(f"Results obtained in {total_time} seconds")
        dash = '-' * 110
        print(dash)
        print('{:<8s}{:<90s}{:>5s}'.format("ID", "Title", "Score"))
        print(dash)
        for docId in top_k_docs:
            docname = doc_info[docId][0]
            print('{:<8s}{:<90s}{:>5.5f}'.format(docId, docname, scores[docId]))
    else:
        print("No matching results found")

        
def processWords(text):
    '''
    Removes all punctuations from the given text.
    
    Parameters:
        text(String): The text to be processed.
    Returns:
        processed_text(List of Strings): The processed text.
        
    '''
    
    split_chars = r'[\u2014\u2013_\-]'
    
    # Splitting a string separated by hyphen and underscore into two words
    text = re.sub(split_chars, ' ', text)
    
    # Removing all other punctuations and then converting to lower case.
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text.split()