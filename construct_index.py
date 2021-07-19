"""
Constructs the inverted index along with helper files.

Command Line Arguments:
    --zoned_index: True if zoned indexing must be used. Set to False by default.
"""

import os.path
import pickle
import time
import argparse
import sys
import math
from os.path import dirname, abspath
from tqdm import tqdm
from bs4 import BeautifulSoup
from helper_module import processWords, createChampionList, strToBool

def addTitleToIndex(title):
    '''
    Extracts each word in the given title, removes unnecessary punctuations, and adds to the temporary index.
    
    Parameters: 
        title (String): The title name to be processed.
    Returns: 
        temp_index (dictionary): The temporary inverted index constructed for the title
        
    '''
    temp_index = {}
    for word in processWords(title):
        if word != '':
            temp_index[word] = temp_index.get(word, 0) + 1
    return temp_index

def addDocToIndex(text):
    '''
    Extracts each word in the given text, removes unnecessary punctuations, and adds to the temporary index.
    
    Parameters: 
        text (String): The text to be processed.
    Returns: 
        temp_index (dictionary): The temporary inverted index constructed for the text
    
    '''
    temp_index = {}
    for line in text.splitlines():
        for word in processWords(line):
            
            # Add every valid word to the temporary index
            if word != '':
                temp_index[word] = temp_index.get(word, 0) + 1
                
    return temp_index



def mergeDocToIndex(main_index, doc_info, doc_id, temp_index, isTitle = False):
    '''
    Merges the temporary index with the inverted index and updates normalization factor for each doc.
    
    Parameters: 
        main_index (dictionary): The inverted index being constructed.
        doc_info (dictionary): The additional document info containing id, name and normalization factor.
        doc_id (String): The ID of the document used in the creation of the temporary index.
        temp_index (dictionary): The index constructed for each doc.
        isTitle (Boolean): True if the index construction is done for the title.
    Returns:
        None
        
    '''
    doc_norm_factor = 0
    for (word, tf) in temp_index.items():
        
        # Add the new document ID and tf to the main index for each word in the temporary index
        main_index[word] = main_index.get(word, [0, {}])
        main_index[word][0] += 1
        main_index[word][1][doc_id] = tf
        
        # Update normalization factor using the logarithmic version of the tf
        doc_norm_factor += (1 + math.log(tf, 10)) ** 2
        
    # Update normalization factor for either content or title
    if not isTitle:
        doc_info[doc_id][1] = doc_norm_factor ** 0.5
    else:
        doc_info[doc_id][2] = doc_norm_factor ** 0.5


        
def createIndexFromCorpus(path_to_corpus, is_zoned_index):
    '''
    Builds the inverted index from the corpus.
    
    Parameters:
        path_to_corpus (String): The path to the "corpus" folder.
        is_zoned_index (bool): True if zoned inverted index must be constructed.
    Returns:
        inverted_index (dictionary): The inverted index of the corpus for the document contents.
        title_index (dictionary): The inverted index of the corpus for the document titles.
        doc_info (dictionary): Contains ID, name, and both (content and title) normalization factor for each document.
        
    '''
    inverted_index = {}
    title_index = {}
    doc_info = {}
    docs = []
    for root, dirs, files in os.walk(path_to_corpus):
        for filename in tqdm(files, desc = "Parsing html files", ascii = False, ncols = 75, file = sys.stdout):
            
            # Extract all the content (including html tags) of the file into filedata
            with open(os.path.join(root, filename), "r", encoding="utf8") as file:
                filedata = file.read()
            
            # Find all documents based on the <doc> tag
            soup = BeautifulSoup(filedata, 'html.parser')
            docs.extend(soup.find_all('doc'))
     
    print("")
    for doc in tqdm(docs, desc = "Processing Corpus", ascii = False, ncols = 75, file = sys.stdout):

        # Extract id, title, and the contents of each doc
        doc_id = doc["id"]
        doc_name = doc["title"]
        doc_text = doc.get_text()

        # Create temporary indices for content and title, and merge into respective main indices
        doc_info[doc_id] = [doc_name, 0, 0]
        temp_index = addDocToIndex(doc_text)
        mergeDocToIndex(inverted_index, doc_info, doc_id, temp_index)

        if(is_zoned_index):
            temp_title_index = addTitleToIndex(doc_name)
            mergeDocToIndex(title_index, doc_info, doc_id, temp_title_index, isTitle = True)
    
    print("")
                
    # Create champions list for inverted_index with a maximum of 100 documents in the posting list for each word
    if(is_zoned_index):
        for word in inverted_index:
            inverted_index[word][1] = createChampionList(inverted_index[word][1], max_docs = 100)
            
    return (inverted_index, title_index, doc_info)

    
if __name__ == "__main__":
    
    # Command Line Args for whether to include title scores or not
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoned_index", type = strToBool, default = False, help = "True for zoned inverted index. False for basic inverted index")
    args = vars(parser.parse_args())
    if(args["zoned_index"] == None):
        print("Incorrect command line arguments. Enter True/False.")
        sys.exit()
    
    path_to_corpus = os.path.join(os.getcwd(), 'corpus')
    
    if(args["zoned_index"]):
        path_to_index = os.path.join(os.getcwd(),'index_files','zoned')
    else:
        path_to_index = os.path.join(os.getcwd(),'index_files','basic')
    
    # Construct the inverted index
    start = time.time()
    (inverted_index, title_index, doc_info) = createIndexFromCorpus(path_to_corpus, args["zoned_index"])
    end = time.time()
    
    print("\nInverted index constructed in: {0:.2f} seconds".format(end - start))
    print("Size of corpus:", len(doc_info), "documents")
    print("Vocabulary size:", len(inverted_index), "words")
    
    # Write all three dictionaries to the disk
    with open(os.path.join(path_to_index, "inverted_index"), "wb") as file:
        pickle.dump(inverted_index, file)
    with open(os.path.join(path_to_index, "doc_info"), "wb") as file:
        pickle.dump(doc_info, file)
    if(args["zoned_index"]):
        with open(os.path.join(path_to_index, "title_index"), "wb") as file:
            pickle.dump(title_index, file)
        
    print("Inverted Index saved to disk")