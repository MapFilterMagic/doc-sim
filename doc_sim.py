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

# import argparse # For command-line argument parsing

import nltk.stem  # For English word stemmer

# import scipy as sp
# import numpy as np


# Class Name: StemmedTfidfVectorizer
# Member Variables: english_stemmer -- English word-stemmer
# Description: Adds English Stemming functionality to TF-IDF vectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    # Intantiate an English SnowballStemmer
    english_stemmer = nltk.stem.SnowballStemmer('english')

    # Function Name: build_analyzer
    # Description:
    # Parameters: self -- reference to this object
    # Return Value: A list of english word stems
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in anaylzer(doc))

# Function Name: parse_file(args.file)
# Description: Control flow responsibile for instantiating various classes,
#              handling command-line arguments, file-processing, and operating
#              on the text.
# Parameters: args.file -- argment list consistant 
# Return Value: file_list -- list containing the text from target and comparison
#                            files
def parse_file(args.file):
    file_list = []  # List of all files (target and comparison)

    # Go through all files passed in arguments
    for f in args.file:
        # Go through all lines in current file
        for line in f:
            # Seperate words by whitespace and add to collective list
            line_list = [elt.strip() for elt in line.split()]
            file_list.append(line_list)

    return file_list

# Function Name: main()
# Description: Control flow responsibile for instantiating various classes,
#              handling command-line arguments, file-processing, and operating
#              on the text.
# Parameters: none
# Return Value: none
def main():
    # Instantiate the TD-IDF English-stemmed vectorizer
    vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english',
                                        decode_error='ignore')
    parser = argparse.ArgumentParser()
    parse.add_argument('target file', type=argsparse.FileType('r'), nargs=1)
    parser.add_argument('comparison file(s)', type=argsparse.FileType('r'), nargs='+')

    # Learn vocabulary from the target file 
    vectorizer.fit_transform(file_list[0])


if __name__ == '__main__':
    main()
