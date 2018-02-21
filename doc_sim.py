#
# Name: Tony Murillo
# Filename: DocSimilarity
# Date: 12 February, 2018
# Description: A command-line program that utilizes the KMeans clustering
#              algorithm, TF-IDF vectorizer, and NTLK Stemmer to find related
#              posts and calculate document similarity
# Credits: For codeblocks & techniques via the book "Building Machine Learning
#          Systems with Python" by Willi Richert and Luis Pedro Coelho
#
# Under the MIT License
#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from math import sqrt

import argparse
import nltk.stem

import scipy as sp
import numpy as np


class StemmedTfidfVectorizer(TfidfVectorizer):
    """Implement an English word-stemmer within a TF-IDF vectorizer.

    Overides the tokenzation function to use the nltk word-stemmer.

    """

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization.

        Include an English word-stemmer during pre-processing and tokenization.

        """
        # Intantiate an English SnowballStemmer
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(TfidfVectorizer, self).build_analyzer()

        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def parse_text(file_args):
    """Build a list of appended text file data.

    Iterate over all file arguments and read each file. Within each text file,
    strip all newlines and replace them with a single whitespace. Append the
    formatted text to the file list to return. Able to parse 1 or more text
    files.

    In the case of 1 file (most often, the target file), return a list with a
    single formatted post, free of newlines.

    In the case of 2 or more files, return a list
    file, return a list with a single string of the

    """
    # List of all files (target and comparison)
    file_list = []

    # Go through 1 or more files passed in as arguments via file_args,
    # remove strip newlines, and append it to the file_list
    for f in file_args:
        text = f.read().replace('\n', ' ')
        file_list.append(text)

    return np.array(file_list)


def est_clust_amt(shape):
    """Estimate number of clusters to use in KMeans algorithm.

    Approximate number of clusters is equivalent to the square root of half of
    all datapoints/samples.

    """
    # Half of all datapoints
    HALF = 0.5

    return int(sqrt((shape[0] * HALF)))


def main():
    """Responsbile general control flow and handling command line arguments.

    Build argparser arguments and parse text files and vectorize them. Then
    use KMeans clustering to predict the target file

    """
    # Words with counts less than threshold to be ignored
    thresh = 2

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

    # Learn vocabulary from the comparison file(s)
    vectorized = vectorizer.fit_transform(comparison)

    # Number of clusters is approx the square root of half of all datapoints
    num_clust = est_clust_amt(vectorized.shape)

    # Assigned random_state argument of KMeans()
    state = 3

    km = KMeans(n_clusters=num_clust, n_init=1, verbose=0, random_state=state)
    km.fit(vectorized)

    # Predict the target file
    target_vectorized = vectorizer.transform(target)
    target_label = km.predict(target_vectorized)

    # Find posts in the same cluster via their indices
    sim_i = np.nonzero(km.labels_ == target_label)[0]

    # Will hold posts sorted by their similarity score in comparison to the
    # target file
    similar_files = []

    # Build similarity score and post list
    for i in sim_i:
        dist = sp.linalg.norm((target_vectorized - vectorized[i]).toarray())
        similar_files.append((dist, comparison[i]))

    similar_files = sorted(similar_files)

    print('=================================================================' +
          '===\n')
    print(similar_files[0])
    print('\n===============================================================' +
          '====\n')


if __name__ == '__main__':
    """ Will only execute if module is ran directly."""
    main()
