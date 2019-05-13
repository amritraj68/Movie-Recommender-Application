#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np  # linear algebra


# In[162]:


import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# In[163]:


import ast


# In[164]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[165]:


data = pd.read_csv('movies.csv', encoding='ISO-8859-1')

data = data[~data['overview'].isna()]


# In[166]:

# Shows the number of features present in the movies.csv dataset file
data.columns


# In[167]:


# which rows actually have genres and not a blank list
has_genres_mask = data['genres'] != '[]'


# In[168]:


genres = data['genres'][has_genres_mask]


# In[169]:


#genres.head(10)


# In[170]:

#Function to extract only the "name " field of genre from the genres feature
def to_labels(genres_list):
    genres_list = ast.literal_eval(genres_list)
    return [g['name'] for g in genres_list]


# Bringing the genres to 8 genres
# 
# Action,
# Adventure
# Fantasy
# Science
# Fiction
# Crime
# Drama
# Thriller

# In[171]:


genres_strings = genres.apply(to_labels)


# In[172]:


genres_strings.head()


# In[173]:


labeler = MultiLabelBinarizer()


# In[174]:


labeler.fit(genres_strings)


# In[175]:


labeler.classes_


# In[176]:


X = data['overview'][has_genres_mask]
y = labeler.transform(genres_strings)
#X= data.overview.fillna('')
#y=data.fillna('')

X.shape, y.shape


# 4758 movies distributed among 20 genres

# In[177]:


pd.DataFrame(y, columns=labeler.classes_).corr() 
#Compute pairwise correlation of columns, excluding NA/null values.Minimum number of observations required per pair of columns to have a valid result.


# In[178]:


import matplotlib.pyplot as plt


# In[179]:


plt.imshow(pd.DataFrame(y).corr())
#ou may be wondering why the x-axis ranges from 0-15 and the y-axis from 0. 
#If you provide a single list or array to the plot() command, matplotlib assumes it is a sequence of y values, and automatically generates the x values for you. Since python ranges start with 0,
#the default x vector has the same length as y but starts with 0. Hence the x data are [0,1,2,3].


# In[180]:


top = sorted(list(zip(y.sum(axis=0), labeler.classes_)))[::-1]
# re-arrange the array of genres by column value greatest to least


# In[181]:


top


# In[182]:


#getting the top genres from the "top"
top_genres =sorted([t[1] for t in top][1:20])
top_genres


# In[183]:


#Example : Drama class is not present in the top_genres, so it will be ignored
top_labeler = MultiLabelBinarizer(classes=top_genres)
top_labeler.fit(genres_strings)
top_labeler.transform([['this is a' ,'Horror']])


# transforming the genres_string to contains only the top genres i.e 
# it will excluded the 'Animation', 'Documentary', 'Drama', 'Fantasy', 'Foreign', 'History', 'Music', 'Mystery', 'TV Movie', 'War', 'Western']
# 
# 

# In[184]:


y = top_labeler.transform(genres_strings)


# In[185]:


len(y.sum(axis=1)!=0), sum(y.sum(axis=1)!=0)


# In[186]:


no_labels_mask = y.sum(axis=1)==0
sum(no_labels_mask),len(no_labels_mask)


# In[187]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(
    X[~no_labels_mask], y[~no_labels_mask], test_size=0.33, random_state=42)


# In[189]:


from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[190]:


counter = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)


# In[191]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


# In[192]:


pipe = Pipeline(
    [
        ('text_transform', counter),
        ('predictor', MLPClassifier(warm_start=True, max_iter=5, hidden_layer_sizes=(10)))
#         ('predictor', RandomForestClassifier(class_weight='balanced'))
    ])


# In[193]:


for i in range(200):
    pipe.fit(X_train, y_train)
    #print('epoc {0}, train {1:.3f}, test {2:.3f}'.format(i, 
                                                       #  pipe.score(X_train, y_train),
                                                         #pipe.score(X_test, y_test)))
    


# In[194]:


pipe.score(X_train,y_train)


# # Fit a naive bayes model to the training data.
# # This will train the model using the word counts we computer, and the existing classifications in the training set.
# nb = MultinomialNB()

# In[195]:


pipe.score(X_test, y_test)


# In[196]:


def get_genres(text):
    print(pipe.predict_proba([text]))
    genres_list = top_labeler.classes_[pipe.predict([text]).ravel().astype('bool')]
    return genres_list

#get_genres('It was a fantactic movie that was horrifying and bit funny as well')





