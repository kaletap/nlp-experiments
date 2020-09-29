import os

import pandas as pd


def read_sentiment_treebank(data_path):
    # Paths
    text_path = os.path.join(data_path, 'datasetSentences.txt')
    sentiment_path = os.path.join(data_path, 'sentiment_labels.txt')
    split_path = os.path.join(data_path, 'datasetSplit.txt')
    dictionary_path = os.path.join(data_path, 'dictionary.txt')
    # Reading csvs
    text_df = pd.read_csv(text_path, sep='\t')
    sentiment_df = pd.read_csv(sentiment_path, sep='|')
    split_df = pd.read_csv(split_path, sep=',')

    dictionary_df = pd.read_csv(dictionary_path, sep='|', header=None)
    dictionary_df.columns = ['phrase', 'id']

    # Extracting sentiment for each sentence (excluding partial phrases)
    id_to_sentiment = {id_: sentiment for id_, sentiment in
                       zip(sentiment_df['phrase ids'], sentiment_df['sentiment values'])}
    phrase_to_sentiment = {sentence: id_to_sentiment[id_] for sentence, id_ in
                           zip(dictionary_df['phrase'], dictionary_df['id'])}
    sentiments = [phrase_to_sentiment.get(phrase) for phrase in text_df['sentence']]
    text_df['sentiment'] = sentiments
    text_df = pd.merge(text_df, split_df, on='sentence_index')
    return text_df
