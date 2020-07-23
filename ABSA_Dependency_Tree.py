import textblob
from textblob import TextBlob
import numpy as np
from gensim.models.fasttext import FastText
import nltk 
from nltk.stem import WordNetLemmatizer
import stanza 


nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

import stanza 
from sentiment_config import * 
from model import get_word_polarity, get_sentiment, get_senti_coeff_indic, sentiment_coeff

stanza.download('en')
stanza.download('hi')
stanza.download('ta')
stanza.download('te')
stanza.download('mr')


# function to find polar words in a sentence using dependency tree
# take the aspect term, a sentence from the article and language_code 
def polar_dependency_tree(aspect_term, article_line, lang_code): 
    # handling the different languages for which the module can work 
    if lang_code == 'en':
        nlp = stanza.Pipeline('en')
    if lang_code == 'hi':
        nlp = stanza.Pipeline('hi')
    if lang_code == 'mr':
        nlp = stanza.Pipeline('mr')
    if lang_code == 'te':
        nlp = stanza.Pipeline('te')
    if lang_code == 'ta':
        nlp = stanza.Pipeline('ta')
    
    # using stanza pipeline to perform dependency parsing
    article_line = nlp(article_line)
    sentence = article_line.sentences[0]
    # list of dependency relations which can have polar words 
    relations = ['acl', 'advcl', 'advmod', 'amod', 'xcomp', 'neg', 'parataxis', 'ccomp']
    # parts of speech tags that can have polar words 
    pos_tags = ['VERB', 'ADJ']
    entity_present = False
    
    # checking if the entity is present or not.
    for word in sentence.words:
        if word.text == aspect_term:
            entity = word
            entity_present = True
    if entity_present == False:
        return []
    
    # list of polar words with respect to the entity
    polar_words = []
    # Case 1: our entity is the root term, so checkout all the child nodes 
    if entity.deprel == 'root':
        for word in sentence.words:
            if (word.head == int(entity.id) and (word.deprel in relations or word.upos in pos_tags)):
                polar_words.append(word.text)
    # Case 2: our entity is not the root term, so traceback the path of the tree from entity to root.
    # also check all the other child nodes after finding the root 
    else:
        for word in sentence.words:
            if word.deprel == 'root':
                root_node = word
        # checking the child nodes of the root 
        for word in sentence.words:
            if (word.head == int(root_node.id) and (word.deprel in relations or word.upos in pos_tags)):
                polar_words.append(word.text)
        # tracing the path from entity to the root node 
        current_word = entity
        while(current_word.deprel != 'root'):
            current_word = sentence.words[current_word.head - 1]
            if (current_word.deprel in relations or current_word.upos in pos_tags):
                polar_words.append(current_word.text)
    
    return list(set(polar_words)) 


def get_polarity_dep_tree(aspect_term, sentences, lang_code,model=None ):
	polarity = 0
	for sentence in sentences:
		polar_words = polar_dependency_tree(aspect_term, sentence, lang_code)
		if len(polar_words) > 0:
			if lang == 'en':
				senti_vector = sentiment_coeff(polar_words)
			else:
				senti_vector = get_senti_coeff_indic(polar_words,lang_code, model)

			sentiment = sum(senti_vector)
			polarity = polarity + sentiment
		else:
			polarity = polarity + 0 

	if polarity > 0:
		return 1
	elif polarity == 0:
		return 0
	else:
		return -1 