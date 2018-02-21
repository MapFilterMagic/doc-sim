#
# Name: Tony Murillo
#
# Filename: DocSimilarity
#
# Date: 12 February, 2018
#
# Description: A command-line program that utilizes the KMeans clustering
#              algorithm, TF-IDF vectorizer, and NTLK Stemmer to find related
#              posts and calculate document similarity
#
# Credits: Various codeblocks/techniques via the book "Building Machine Learning
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
    file_list = []  # List of all files (target and comparison)

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
    HALF = 0.5  # Half of all datapoints

    return int(sqrt((shape[0] * HALF)))


def main():
    """Responsbile general control flow and handling command line arguments.

    """
    thresh = 2  # Words with counts less than threshold to be ignored

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

    state = 3  # Assigned random_state argument of KMeans()

    km = KMeans(n_clusters=num_clust, n_init=1, verbose=1, random_state=state )
    km.fit(vectorized)

    target_vectorized = vectorizer.transform(target)
    target_label = km.predict(target_vectorized)

    sim_i = np.nonzero(km.labels_ == target_label)[0]

    similar_files = []

    for i in sim_i:
        dist = sp.linalg.norm((target_vectorized - vectorized[i]).toarray())
        similar_files.append((dist, comparison[i]))

    similar_files = sorted(similar_files)

    print(similar_files[0])


if __name__ == '__main__':
    main()
