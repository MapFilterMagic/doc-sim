# Name: Tony Murillo
#
# Filename: DocSimilarity
#
# Date: 12 February, 2018
#
# Description: A command-line program that utilizes the KMeans clustering
# algorithm, TF-IDF vectorizer, and q the NTLK Stemmer to find related posts
# and calculate document similarity

from sklearn.feature_extraction.text import TfidfVectorizer

# import os
# import sys # For file-path operations

import argparse  # For command-line argument parsing

import nltk.stem  # For English word stemmer

# import scipy as sp
# import numpy as np


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
# Return Value: file_list -- list containing the text data from arg
def parse_text(file_args):
    file_list = []  # List of all files (target and comparison)

    # Go through 1 or more files  passed in arguments
    for f in file_args:
        # Go through all lines in current file
        for line in f:
            # Seperate words by whitespace and add to collective list
            line_list = [elt.strip() for elt in line.split()]
            file_list.append(line_list)
            flat_file_list = [y for x in file_list for y in x]

    return flat_file_list


# Function Name: main()
# Description: Control flow responsibile for instantiating various classes,
#              handling command-line arguments, file-processing, and operating
#              on the text.
# Parameters: none
# Return Value: none
def main():
    thresh = 10  # Words with counts less than threshold to be ignored

    # Instantiate the TD-IDF English-stemmed vectorizer
    vectorizer = StemmedTfidfVectorizer(min_df=thresh, stop_words='english',
                                        decode_error='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('target_file', type=argparse.FileType('r'), nargs=1)
    parser.add_argument('comparison_file', type=argparse.FileType('r'),
                        nargs='+')
    args = parser.parse_args()

    # Parse text data for target post and comparison files
    target = parse_text(args.target_file)
    comparison = parse_text(args.comparison_file)

    # print(comparison)

    # Learn vocabulary from the target file
    vectorized = vectorizer.fit_transform(target)

    km = KMeans(n_clusters
    print(vectorized.shape)
    #print(comparison.shape)


if __name__ == '__main__':
    main()
