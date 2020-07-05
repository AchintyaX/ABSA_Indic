import textblob 
from textblob import TextBlob

def aspect_term_location(aspect_term, article, lang_code):
    if lang_code == 'en':
        aspect_term = spacy_tokenizer(aspect_term)
    location_points = []
    start_point = article.index(aspect_term[0])
    location_points.append(start_point)
    if len(aspect_term) > 1:
        end_point = article.index(aspect_term[len(aspect_term)-1])
        location_points.append(end_point)
    
    return location_points 


def word_distance(aspect_term, article, lang_code):
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
		polarity = TextBlob(article[i]).polarity
		distance = i 
		tot_polarity_start = tot_polarity_start + polarity*(distance/start_point)

	print(tot_polarity_start)

	end_dist = len(article) - 1 
	for i in range(end_point+1, len(article)):
		polarity = TextBlob(article[i]).polarity 
		distance = end_dist - i
		tot_polarity_end = tot_polarity_end + (distance/end_dist)*polarity

	print(tot_polarity_end)

	tot_polarity = tot_polarity_start + tot_polarity_end
	return tot_polarity
