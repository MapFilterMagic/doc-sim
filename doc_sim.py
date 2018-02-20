#
# Name: Tony Murillo
#
# Filename: DocSimilarity
#
# Date: 12 February, 2018
#
# Description: A command-line program that utilizes the KMeans clustering
# algorithm, TF-IDF vectorizer, and q the NTLK Stemmer to find related posts
# and calculate document similarity
#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  # For clustering

from math import sqrt  # For cluster calculation

# import os
# import sys # For file-path operations

import argparse  # For command-line argument parsing

import nltk.stem  # For English word stemmer


import scipy as sp
import numpy as np


# Class Name: StemmedTfidfVectorizer
# Member Variables: english_stemmer -- English word-stemmer
# Description: Adds English Stemming functionality to TF-IDF vectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):

    # Function Name: build_analyzer
    # Description:
    # Parameters: self -- reference to this object
    # Return Value: A list of english word stems
    def build_analyzer(self):
        # Intantiate an English SnowballStemmer
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(TfidfVectorizer, self).build_analyzer()

        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# Function Name: parse_text(args.file)
# Description: Builds a list of appended text file data
# Parameters: file_args -- file argument list
# Return Value: file_list -- flattended list containing the text data from arg
def parse_text(file_args):
    file_list = []  # List of all files (target and comparison)

    # Go through 1 or more files passed in as arguments via file_args,
    # remove strip newlines, and append it to the file_list
    for f in file_args:
        text = f.read().replace('\n', ' ')
        file_list.append(text)

    return np.array(file_list)


# Function Name: est_clust_amt(vectorized)
# Description: Estimates number of clusters by taking the square root of half
#              of all datapoints or samples.
# Parameters: vectorized -- vectozied text data from which to get sample count
#                           from
# Return Value: estimated number of clusters
def est_clust_amt(shape):
    HALF = 0.5  # Half of all datapoints

    return int(sqrt((shape[0] * HALF)))


# Function Name: main()
# Description: Control flow responsibile for instantiating various classes,
#              handling command-line arguments, file-processing, and operating
#              on the text.
# Parameters: none
# Return Value: none
def main():
    thresh = 1  # Words with counts less than threshold to be ignored

    # Instantiate the TD-IDF English-stemmed vectorizer
    vectorizer = StemmedTfidfVectorizer(min_df=thresh, stop_words='english',
                                        decode_error='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=argparse.FileType('r'), nargs=1)
    parser.add_argument('comparison', type=argparse.FileType('r'),
                        nargs='+')
    args = parser.parse_args()

    # Parse text data for target post and comparison files
    target = parse_text(args.target)
    comparison = parse_text(args.comparison)
    # DEBUGGING -- REMOVE WHEN FINISHED
    # print('target shape:%s' % target.shape)
    # print('target:%s' % target)
    # print('comparison shape:%s' % comparison.shape)
    # print('comparison:%s' % comparison)

    # Learn vocabulary from the comparison file(s)
    vectorized = vectorizer.fit_transform(comparison)

    # Number of clusters is approx the square root of half of all datapoints
    num_clust = est_clust_amt(vectorized.shape)

    # state = 3  # Assigned random_state argument of KMeans()

    km = KMeans(n_clusters=num_clust, n_init=1, verbose=1 )
    km.fit(vectorized)
    # print(vectorized.get_feature_names())

    #new_post = ["Means I don't fuck with you"]

    target_vectorized = vectorizer.transform(target)
    #target_vectorized = vectorizer.transform(new_post)
    target_label = km.predict(target_vectorized)
    # print(target_vectorized.get_feature_names())
    # target_label = km.predict(target_vectorized)
    print("target_label prediction:%s" % target_label)
    #print("target_label shape:", target_label.shape)
    #print('target_label: %s' % target_label)
    #print("target_label:%s" % target_label)

    # print(len(km.labels_))
    #print(len(target_label))

    # print(len(sim_i))

    # print(np.nonzero(km.labels_))
    sim_i = np.nonzero(km.labels_ == target_label)[0]

    # print("sim i")
    # print(sim_i)

    # similar_files = []

    # print("Shape of target:", target_vectorized.shape)
    # print("Shape of comp:", vectorized.shape)
    # for i in sim_i:
    #    dist = sp.linalg.norm((target_vectorized - vectorized[i]).toarray())
    #    similar_files.append((dist, comparison[i]))

    # print("Similar: %s" % similar_files)
    # similar_files = sorted(similar_files)

    # print("Count Similar: %i" %len(similar_files))


if __name__ == '__main__':
    main()
