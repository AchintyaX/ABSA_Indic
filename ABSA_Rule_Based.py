import textblob 
from textblob import TextBlob
from article_preprocessing import tokenize_hin, spacy_tokenizer, remove_hin_stopwords
import numpy as np
from gensim.models.fasttext import FastText 
from model import get_sentiment, get_word_polarity, get_word_polarity_en
from sentiment_config import *

filepath = HINDI_STOPWORDS

# used to find the location points of aspect term in the review
# takes tokenized aspect term and article as input 
def aspect_term_location(aspect_term, article, lang_code):
    location_points = []
    start_point = article.index(aspect_term[0])
    location_points.append(start_point)
    if len(aspect_term) > 1:
        end_point = article.index(aspect_term[len(aspect_term)-1])
        location_points.append(end_point)
    
    return location_points 

# Used to find the aspect term sentiment, for a sentence in the review 
# takes the aspect term, sentence and language code as the input
# pass pos_words and neg_words dictionaty list along with the FastText embeddings model when using Indian languages
# pass the respective model and word lists for a given language.
# The aspect term and sentence need to be tokenized 
def word_distance_based_score(aspect_term, article, lang_code, model=None):
	aspect_locations = aspect_term_location(aspect_term, article, lang_code)

	if len(aspect_locations) == 1:
		start_point = end_point = aspect_locations[0]
	else:
		start_point = aspect_locations[0]
		end_point = aspect_locations[1]

	print(start_point)
	print(end_point)

	tot_polarity_start = 0
	tot_polarity_end = 0 

	for i in range(start_point):
		if lang_code == 'en':
			polarity = get_word_polarity_en(article[i])
		if lang_code =='hi':
			polarity = get_word_polarity(article[i],lang_code, model)
		distance = i 
		tot_polarity_start = tot_polarity_start + polarity*(distance/start_point)

	print(tot_polarity_start)

	end_dist = len(article) - 1 
	for i in range(end_point+1, len(article)):
		if  lang_code == 'en':
			polarity = get_word_polarity_en(article[i])
		if lang_code == 'hi':
			polarity = get_word_polarity(article[i], lang_code, model)
		distance = end_dist - i
		tot_polarity_end = tot_polarity_end + (distance/end_dist)*polarity

	print(tot_polarity_end)

	tot_polarity = tot_polarity_start + tot_polarity_end
	return tot_polarity

	
# Function is used to find the final polarity of the aspect term in an article
# Takes the aspect term, sentences and the language code as input
# pass pos_words and neg_words dictionaty list along with the FastText embeddings model when using Indian languages
# pass the respective model and word lists for a given language.
# the aspect term should be tokenized 
# the sentences mean that a whole article should be broken down into sentences using the sentence
# segmentation 
def aspect_polarity(aspect_term, sentences, lang_code, model=None):
    if lang_code == 'en':
        sentences = [spacy_tokenizer(i) for i in sentences]
    if lang_code == 'hi':
        sentences = [tokenize_hin(i) for i in sentences]
        sentences = [remove_hin_stopwords(i, filepath) for i in sentences]
        print(sentences)
    tot_polarity = 0
    for review in sentences:
        try:
            polarity = word_distance_based_score(aspect_term, review, lang_code, model=model )
            tot_polarity = tot_polarity + polarity
        except ValueError: # for handling sentences which don't have aspect term in the article 
            tot_polarity = tot_polarity + 0 
    
    if tot_polarity > 0: 
        aspect_polarity = 1
    elif tot_polarity == 0:
        aspect_polarity = 0
    else:
        aspect_polarity = -1 

    return aspect_polarity 