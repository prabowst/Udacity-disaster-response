import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import sys
import pickle
import os
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

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
    

def load_data(database_filepath):
    '''
    Load cleaned pandas DataFrame from the ETL pipeline

    Args:
        database_filepath (str): sql path for cleaned disaster data

    Returns:
        X (pandas.Series): selected feature for machine learning model
        Y (pandas.DataFrame): target values for machine learning model
        category_names (pandas.index): categories class names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_cleaned', con = engine)
    df['related'] = df['related'].replace(2, 1)
    df = df.drop('child_alone', axis = 1)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Create clean tokens for messages to fit in machine learning model

    Args:
        text (str): disaster messages to be cleaned through NLP

    Returns:
        clean_tokens (str): clean tokens of words 
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_found = re.findall(url_regex, text)
    for url in urls_found:
        text = text.replace(url, 'urlplaceholder')
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(words).strip() for words in tokens]
    
    return clean_tokens

def build_model():
    '''
    Building a pipeline model after parameter tuning
    Note: The parameters for actual GridSearchCV is commented due to 
          the process might take high runtime to successfully run

    Args:
        None

    Returns:
        grid: machine learning pipeline model; combination of NLP 
              and classifier (AdaBoostClassifier)
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(estimator = AdaBoostClassifier()))
    ])
    
    # parameters = {
    #    'clf__estimator__n_estimators': [50, 100],
    #    'clf__estimator__learning_rate': [1, 1.2]
    # }
    
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [1]
    }
    
    grid = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 3) 
    
    return grid
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performance by printing lines of classification report
    for each categories
    
    Args:
        model: machine learning pipeline model
        X_test (pandas.Series): test feature data
        Y_test (pandas.DataFrame): test target data
        category_names (pandas.index): categories class names

    Returns:
        accuracy (str): text containinng model's accuracy as overall
    '''
    Y_prediction = model.predict(X_test)
    Y_prediction_df = pd.DataFrame(Y_prediction, columns = category_names)
    
    for col in category_names:
        print(str(col) + ' category:')
        print(classification_report(Y_test[col], Y_prediction_df[col]))
        print(str('_____________________________________________________'))
    
    accuracy = (Y_prediction == Y_test).mean().mean()
    
    return print('Accuracy of model: ' + str(round(accuracy*100,2)) + '%')


def save_model(model, model_filepath):
    '''
    Save the trained model to a pickle file
    
    Args:
        model: machine learning pipeline model
        model_filepath (str): model's save location 
        
    Returns:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return None


def main():
    '''
    The entire ML pipeline process:
    
    1) Extract the cleaned data from database
    2) Split the cleaned data into X_train, X_test, Y_train, Y_test
    3) Build and train the model by fitting X_train, Y_train
    4) Predict the model on X_test
    5) Compare the Y_prediction with Y_test using classification report
    6) Save the model using pickle
    
    Args:
        None
        
    Returns:
        None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()