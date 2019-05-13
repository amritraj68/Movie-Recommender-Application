import numpy as np 
import nltk
#nltk.download('stopwords')  
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
from flask import Flask, request, jsonify, render_template
from search import search_r
from classify import get_genres
from Recommend import get_recommendations
from sklearn.feature_extraction.text import CountVectorizer
from Recommend import count_matrix

class c:
    blue = '\033[94m'
    red =  '\033[93m'

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('search.html')


@app.route('/classify', methods=['POST','GET'])
def classify():
    # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
    try:
        _text = request.form.get('classifying')  # get(attr) returns None if attr is not present
        
        if _text:
            dfc= get_genres(_text)
            return render_template('classifyResult.html', myLists=list(dfc))

        else:
            return 'Please go back and enter the text ..', 400  # 400 Bad Request

    except Exception as e:
        print("Error:",e)
        return render_template('search.html')
        print('elseeee')
 
    # Validate and send response

    # Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)

@app.route('/recommend', methods=['POST','GET'])
def recommend():

    # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
    try:
        _recommendText = request.form.get('recommending')  # get(attr) returns None if attr is not present
        
        if _recommendText:
            dfr = get_recommendations(_recommendText)
            return render_template('recommendResult.html', myLists=list(dfr))

        else:
            return 'Please go back and enter the Movie Name ..', 400  # 400 Bad Request

    except Exception as e:
        print("Error:",e)
        return render_template('search.html')
        print('elseeee')
 
    # Validate and send response


@app.route('/process', methods=['POST','GET'])
def process():
    # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
    try:
        _query = request.form.get('username')  # get(attr) returns None if attr is not present
        
        if _query:
            df = search_r(_query)
            return render_template('searchResult.html', tables=[df.to_html(classes='df')], titles=df.columns.values)

        else:
            return 'Please go back and enter your name...', 400  # 400 Bad Request

    except Exception as e:
        print("Error:",e)
        return render_template('search.html')
        print('elseeee')
 
    # Validate and send response

    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

