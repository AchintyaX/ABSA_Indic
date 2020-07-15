import textblob
from textblob import TextBlob
import numpy as np
from gensim.models.fasttext import FastText
import nltk 
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
'''
# retrieving the sentiment vector 
def sentiment_coeff(article):
	sentiment_vector = []
	for word in article:
		token = TextBlob(word).polarity
		sentiment_vector.append(token)

	return sentiment_vector

'''

# retrieving the term frequency vector 
def term_frequency(article):
    term_freq = {}
    for token in article:
        term_freq[token] = article.count(token)
    term_vector = []
    for token in article:
        term_val = term_freq[token]/len(article)
        term_vector.append(term_val)
    return term_vector

def predict_sentiment(article, lang_code, model, pos_words, neg_words):
	term_vector = term_frequency(article)
	if lang_code == 'en':
		senti_vector = sentiment_coeff(article)
	if lang_code == 'hi':
		senti_vector = get_senti_coeff_indic(article, pos_words, neg_words, model)
	term_vector = np.array(term_vector)
	senti_vector = np.array(senti_vector)

	polarity = np.dot(term_vector, senti_vector)

	if polarity > 0:
		num = 1
	elif polarity==0:
		num = 0
	else:
		num = -1 

	return num 
# Getting the sentiment vectors of each sentence 

def get_senti_coeff_indic(article, pos_words, neg_words, model):
    sentiment_vector = []
    for word in article:
        token_pos = get_sentiment(word, pos_words, model)
        token_neg = get_sentiment(word, neg_words, model)
        if token_pos >= token_neg:
            token = token_pos
            if token < 0.40:
                token = 0
        else:
            token = -1*token_neg
            if token > -0.40:
                token = 0 
        sentiment_vector.append(token)
    return sentiment_vector

# updated version of the  get polarity function
def get_sentiment(word, word_list, model):
    max_similarity = 0
    for i in word_list:
        try:
            similarity = model.wv.similarity(word, i['word'])
        except KeyError:
            similarity = 0 
        if i['pos']> i['neg']:
            coeff = i['pos']
        else:
            coeff = i['neg']
        score = similarity*coeff
        if score > max_similarity:
            max_similarity = score
    return  max_similarity

# Getting similar words 
def word_gen(word_array, model, polarity, pos_words_dict, neg_words_dict):
	words = []

	for i in word_array:
		for word in model.wv.most_similar(i):
			words.append(word[0])

	correct_words = []

	if polarity == 1:
		word_list = pos_words_dict

	else:
		word_list = neg_words_dict

	for i in words: 
		if get_sentiment(i, word_list, model) > 0.40:
			correct_words.append(i)

	return correct_words


# Getting word level polarity, return a value between -1 and 1 which reflects  the polarity 
def get_word_polarity(word, pos_words, neg_words, model):
    token_pos = get_sentiment(word, pos_words, model)
    token_neg = get_sentiment(word, neg_words, model)

    if token_pos >= token_neg:
        token = token_pos
        if token < 0.40:
            token = 0
    else:
        token = -1*token_neg
        if token > -0.40 :
            token = 0 
    return token

# Loading polar words 
def get_polar_words(filename):
    with open(filename, encoding="ISO-8859-1") as f:
        words = set(f.read().splitlines())
    return words

pos_words = get_polar_words("resources/pos_lemm.txt")
neg_words = get_polar_words("resources/neg_lemm.txt")


# retrieving the sentiment vector
def sentiment_coeff(article):
    sentiment_vector = []
    flip_polarity = False
    for word in article:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized == "not":
            flip_polarity = True
            score = 0
        elif word_lemmatized in pos_words:
            score = 0.9
        elif word_lemmatized in neg_words:
            score = -0.9
        else:
            score = TextBlob(word).polarity
            if abs(score) < 0.75:
                score = 0
        sentiment_vector.append(score)
    if flip_polarity:
        sentiment_vector = [e * -1 for e in sentiment_vector]
    return sentiment_vector
