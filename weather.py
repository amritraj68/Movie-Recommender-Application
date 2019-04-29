
import numpy as np 
import nltk
nltk.download('stopwords')  
import pandas as pd
import re
import pickle  
from nltk.corpus import stopwords 
from ast import literal_eval
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import sys 
from termcolor import colored, cprint
from sklearn.externals import joblib 

def get_search_results(query):
    query = process_text(query)
    query_matrix = count.transform([query])
    query_tfidf = tfidfconverter.transform(query_matrix)
    sim_score = cosine_similarity(query_tfidf, X)
    sorted_indexes = np.argsort(sim_score).tolist()
    return mergeall_data_different_keys.iloc[sorted_indexes[0][-10:]]

def extract_name_and_character(casts):
    name_character = []
    for cast in casts:
        name_character.append(cast['name'])
        name_character.append(cast['character'])
    return name_character

def extract_required_fields(crews):
    crew_list= []
    for crew in crews:
        if crew['job'] in interested_jobs:
            crew_list.append(crew['name'])
    return crew_list

def extract_names(data):
    name_list = []
    for row in data:
        name_list.append(row['name'])
    return name_list

def combine_all_text(data):
    return data['original_title'] +' '+data['tagline']+' '+data['overview']+' '.join(data['cast_required'])+''.join(data['crew_required'])


def process_text(text):
    text = text.replace("uncredited","")
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))

#print(os.getcwd())
#print(os.listdir(os.getcwd()))
movies_data = pd.read_csv('movies.csv')


credits_data = pd.read_csv ('credits.csv')


mergeall_data_different_keys = pd.merge(movies_data,credits_data, left_on ='id' , right_on = 'movie_id')


mergeall_data_different_keys['cast'] = mergeall_data_different_keys['cast'].apply(literal_eval)


mergeall_data_different_keys['cast_required'] = mergeall_data_different_keys['cast'].apply(extract_name_and_character)
mergeall_data_different_keys['cast_required'].head(5)


mergeall_data_different_keys['crew'][0]


mergeall_data_different_keys['crew'] = mergeall_data_different_keys['crew'].apply(literal_eval)

interested_jobs = ['Director', 'Writer', 'Producer']

mergeall_data_different_keys['crew_required'] = mergeall_data_different_keys['crew'].apply(extract_required_fields)

mergeall_data_different_keys['crew_required'].head(5)

features_common = ['production_companies',
                  'genres',
                  'keywords']

for feature in features_common:
    print(feature,mergeall_data_different_keys[feature][0])


for feature in features_common:
    mergeall_data_different_keys[feature] = mergeall_data_different_keys[feature].apply(literal_eval)
    mergeall_data_different_keys['{}_filtered'.format(feature)] = mergeall_data_different_keys[feature].apply(extract_names)
    print(mergeall_data_different_keys['{}_filtered'.format(feature)][0])

#mergeall_data_different_keys['overview'].describe()

mergeall_data_different_keys['overview'] = mergeall_data_different_keys['overview'].fillna('')

#mergeall_data_different_keys['title_x'].describe()

#mergeall_data_different_keys['tagline'].describe()

mergeall_data_different_keys['tagline'] = mergeall_data_different_keys['tagline'].fillna('')


mergeall_data_different_keys['all_text'] = mergeall_data_different_keys.apply(combine_all_text, axis = 1)

mergeall_data_different_keys['all_text'][0]

stop_words = stopwords.words('english')

mergeall_data_different_keys['all_text'] = mergeall_data_different_keys['all_text'].apply(process_text)
mergeall_data_different_keys['all_text'][0]

english_stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

count = CountVectorizer(analyzer = stemming)

count_matrix = count.fit_transform(mergeall_data_different_keys['all_text'])
 
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(count_matrix).toarray()  

    
def search_r(s):
    search_word=s.lower()
    #text = colored(string, 'red', attrs=['reverse', 'blink']) 
    movies = get_search_results(search_word)
    movies2=movies[['original_title','all_text']].copy()
  
    return movies2


"""
joblib.dump(count,'count.pkl')
joblib.dump(tfidfconverter,'tfidfconverter.pkl')
joblib.dump(X,'traindf.pkl')

fields=['original_title','all_text','tagline']
mergeall_data_different_keys[fields].to_pickle('movies.pkl') 


  #for i in movies2.index:
    #    if search_word in movies2.loc[i,'all_text']:
    #        movies2.loc[i,'all_text']=movies2.loc[i,'all_text'].replace(search_word, '<span style="color: red">{}</span>'.format(search_word))


"""


