#import statements
import csv
import glob
import pandas as pd
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from summa.summarizer import summarize


def text_summarize():
    newsfile = 'C:/Users/Dante/PycharmProjects/CIP/Datasets/Real news/'
    outfile = open("C:/Users/Dante/PycharmProjects/CIP/Datasets/Real news/real_summ.csv", "w+", newline='')
    count=0

    for files in glob.glob(newsfile +"*.txt"):
        print(files)
        count = count+1
        infile = open(files, errors='ignore')
        text = infile.read()
        res = summarize(text, ratio=0.2)
        temp = ""
        for line in res:
            temp += line.rstrip('\n')
        CSVWriter = csv.writer(outfile)
        CSVWriter.writerow([str(temp)])

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def rem_punctuation(text):
    punc_removed = text.translate(str.maketrans('', '', string.punctuation))
    return punc_removed

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


def preprocess():
    sarcastic_datafile = "C:/Users/Dante/PycharmProjects/CIP/Datasets/Real news/real_summ.csv"
    real_datafile = "C:/Users/Dante/PycharmProjects/CIP/Datasets/news/summ.csv"
    s_texts = pd.read_csv(sarcastic_datafile, encoding='latin1', header=None)
    r_texts = pd.read_csv(real_datafile, encoding='latin1', header=None)
    #texts = texts[:200]
    #labels = texts.iloc[:,0]
    #raw_text = texts.iloc[:,1]
    #print(raw_text)

    outfile = "C:/Users/Dante/PycharmProjects/CIP/Datasets/processed_summ.csv"
    f_out = open(outfile, mode='w+', newline='')
    write = csv.writer(f_out, quotechar='"')
    write.writerow(['Headline', 'Label'])
    for index, row in s_texts.iterrows():
        text = row[0]
        if isinstance(text, float):
            continue
        html_free = strip_html(text)
        words = nltk.word_tokenize(html_free)
        words = normalize(words)
        stems, lemmas = stem_and_lemmatize(words)
        normalized = " ".join(lemmas)
        if(normalized != ''):
            write.writerow(['{}'.format(normalized), '1'])

    for index, row in r_texts.iterrows():
        text = row[0]
        if isinstance(text, float):
            continue
        html_free = strip_html(text)
        words = nltk.word_tokenize(html_free)
        words = normalize(words)
        stems, lemmas = stem_and_lemmatize(words)
        normalized = " ".join(lemmas)
        if(normalized != ''):
            write.writerow(['{}'.format(normalized), '0'])

preprocess()
