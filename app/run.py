import json
import plotly
import pandas as pd

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from joblib import dump, load
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def tokenize(text):
    '''
    Create clean tokens for messages

    Args:
        text (str): disaster messages to be cleaned through NLP

    Returns:
        clean_tokens (str): clean tokens of words 
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(words).strip() for words in tokens]

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_cleaned', con = engine)
df['related'] = df['related'].replace(2, 1) # Replace related 2 with 1
df = df.drop('child_alone', axis = 1) # Drop the child_alone feature since it contains only 0s

# load model
model = load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,4:].sum().sort_values(ascending = False)*100/len(df)
    category_counts_label = category_counts.index.tolist()
    
    category_top5 = df.iloc[:,4:].sum().sort_values(ascending = False)[:5]
    category_top5_label = category_top5.index.tolist()
    
    irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                     'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts,
                    marker_color = 'rgb(55, 83, 109)'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Genre'
                }
            }
        },
        {
            'data': [
                Bar(
                    x = category_counts_label,
                    y = category_counts,
                    marker_color = 'rgb(55, 83, 109)'
                )
            ],
            'layout': {
                'title': 'Percentage of Message Categories',
                'yaxis': {
                    'title': 'Percentage (%)'
                },
                'xaxis': {
                    'tickangle': -35
                }
            }
        },
        {
            'data': [
                Pie(
                    labels = category_top5_label,
                    values = category_top5,
                    marker_colors = irises_colors,
                )
            ],
            'layout': {
                'title': 'Top 5 Categories',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()