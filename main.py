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
from weather import search_r

class c:
    blue = '\033[94m'
    red =  '\033[93m'

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('search.html')


@app.route('/process', methods=['POST','GET'])
def process():
    # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
    try:
        _query = request.form.get('username')  # get(attr) returns None if attr is not present
        
        if _query:
            df = search_r(_query)
            return render_template('search2.html',  tables=[df.to_html(classes='df')], titles=df.columns.values)

        else:
            return 'Please go back and enter your name...', 400  # 400 Bad Request

    except Exception as e:
        print("Error:",e)
        return render_template('search.html')
        print('elseeee')
 
    # Validate and send response
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

