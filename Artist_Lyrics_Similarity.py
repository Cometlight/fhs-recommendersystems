import os
import numpy as np
import scipy.spatial.distance as scidist      # import distance computation module from scipy package
import Helper_IO as io

ARTISTS_FILE = "./data/UAM_artists.csv"
INPUT_LYRICS_DIRECTORY = "./data/crawls_lyrics/"
OUTPUT_TFIDF_FILE = "./data/tfidfs.txt"            # file to store term weights
OUTPUT_TERMS_FILE = "./data/terms.txt"             # file to store list of terms (for easy interpretation of term weights)
OUTPUT_SIMS_FILE = "./data/AAM.txt"               # file to store similarities between items

# Stop words used by Google
STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

if __name__ == '__main__':
    text_content = {} # dictionary to hold tokenized lyrics of each artist
    terms_df = {} # dictionary to hold document frequency of each term in corpus
    term_list = [] # list of all terms

    artists = io.read_file(ARTISTS_FILE)

    # for i in range(0, len(artists)):
    for i in range(0, 19):
        print "Processing lyrics of artist {} of {}".format(i, len(artists))
        file_name = INPUT_LYRICS_DIRECTORY + str(i) + ".txt"
        if os.path.exists(file_name):
            content = open(file_name, 'r').read()

            content_casefolded = content.lower() # Case-folding -> convert to all lower case

            tokens = content_casefolded.split() # Tokenize stripped content at white space characters

            tokens_alnum = filter(lambda t: t.isalnum(), tokens) # Remove all tokens containing non-alphanumeric characters

            tokens_filtered_stopped = filter(lambda t: t not in STOP_WORDS, tokens_alnum) # Remove words from the stop word list

            text_content[i] = tokens_filtered_stopped
            print "File " + file_name + " --- total tokens: " + str(len(tokens)) + "; after filtering and stopping: " + str(len(tokens_filtered_stopped))
        else:
            print "{}'s lyrics file ({}) does not exist! Skipping..".format(artists[i], file_name)
    
    # Start computing term weights, in particular, document frequencies and term frequencies.

    # Iterate over all (key, value) tuples from dictionary just created to determine document frequency (DF) of all terms
    for aid, terms in text_content.items():
        # convert list of terms to set of terms ("uniquify" words for each artist/document)
        for term in set(terms):                         # and iterate over all terms in this set
            # update number of artists/documents in which current term t occurs
            if term not in terms_df:
                terms_df[term] = 1
            else:
                terms_df[term] += 1

    # Compute number of artists/documents and terms
    no_artists = len(text_content.items())
    no_terms = len(terms_df)
    print "Number of artists in corpus: " + str(no_artists)
    print "Number of terms in corpus: " + str(no_terms)

    # You may want (or need) to perform some kind of dimensionality reduction here, e.g., filtering all terms
    # with a very small document frequency.
    # ...
    # TODO


    # Dictionary is unordered, so we store all terms in a list to fix their order, before computing the TF-IDF matrix
    for term in terms_df.keys():
        term_list.append(term)

    # Create IDF vector using logarithmic IDF formulation
    idf = np.zeros(no_terms, dtype=np.float32)
    for i in range(0, no_terms):
        idf[i] = np.log(no_artists / terms_df[term_list[i]])
        print term_list[i] + ": " + str(idf[i])

    # Initialize matrix to hold term frequencies (and eventually TF-IDF weights) for all artists for which we fetched HTML content
    tfidf = np.zeros(shape=(no_artists, no_terms), dtype=np.float32)

    # Iterate over all (artist, terms) tuples to determine all term frequencies TF_{artist,term}
    terms_index_lookup = {}         # lookup table for indices (for higher efficiency)
    for a_idx, terms in text_content.items():
        print "Computing term weights for artist " + str(a_idx)
        # You may want (or need) to make the following more efficient.
        for t in terms:                     # iterate over all terms of current artist
            if t in terms_index_lookup:
                t_idx = terms_index_lookup[t]
            else:
                t_idx = term_list.index(t)      # get index of term t in (ordered) list of terms
                terms_index_lookup[t] = t_idx
            tfidf[a_idx, t_idx] += 1        # increase TF value for every encounter of a term t within a document of the current artist

    # Replace TF values in tfidf by TF-IDF values:
    # copy and reshape IDF vector and point-wise multiply it with the TF values
    tfidf = np.log1p(tfidf) * np.tile(idf, no_artists).reshape(no_artists, no_terms)

    # Storing TF-IDF weights and term list
    print "Saving TF-IDF matrix to " + OUTPUT_TFIDF_FILE + "."
    np.savetxt(OUTPUT_TFIDF_FILE, tfidf, fmt='%0.6f', delimiter='\t', newline='\n')

    print "Saving term list to " + OUTPUT_TERMS_FILE + "."
    with open(OUTPUT_TERMS_FILE, 'w') as f:
        for t in term_list:
            f.write(t + "\n")

    # Computing cosine similarities and store them
#    print "Computing cosine similarities between artists."
    # Initialize similarity matrix
    sims = np.zeros(shape=(no_artists, no_artists), dtype=np.float32)
    # Compute pairwise similarities between artists
    for i in range(0, no_artists):
        print "Computing similarities for artist " + str(i)
        for j in range(i, no_artists):
#            print tfidf[i], tfidf[j]
            cossim = 1.0 - scidist.cosine(tfidf[i], tfidf[j])
            # If either TF-IDF vector (of i or j) only contains zeros, cosine similarity is not defined (NaN: not a number).
            # In this case, similarity between i and j is set to zero (or left at zero, in our case).
            if not np.isnan(cossim):
                sims[i,j] = cossim
                sims[j,i] = cossim

    print "Saving cosine similarities to " + OUTPUT_SIMS_FILE + "."
    np.savetxt(OUTPUT_SIMS_FILE, sims, fmt='%0.6f', delimiter='\t', newline='\n')