import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from nltk.tokenize import word_tokenize
import datasets
from gensim.models import FastText

def load_data():
    data = datasets.load_dataset('ucirvine/sms_spam')
    data.set_format(type='pandas')
    df = data['train'].to_pandas()
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)
    return df

def get_text_corpus(texts: Iterable[list]) -> list:
    corpus = [word for text in texts for word in text]
    return list(set(corpus))

def prepare_data(df):
    corpus = get_text_corpus(df['sms'].values)
    X_train, X_test, y_train, y_test = train_test_split(df.sms.values, df.label.values, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, corpus

def build_embedding_dictionary(corpus: list):
    EMB_DIM = 50
    ft = FastText(vector_size=EMB_DIM, window=3, min_count=1)
    ft.build_vocab(corpus_iterable=[corpus])
    ft.train(corpus_iterable=[corpus], total_examples=len(corpus), epochs=10)
    return {word: ft.wv[word] for word in corpus}

def convert_to_embeddings(X: np.ndarray, emb_dict: dict) -> np.ndarray:
    return [np.array([emb_dict.get(word, np.zeros(50)) for word in sample]) for sample in X]
