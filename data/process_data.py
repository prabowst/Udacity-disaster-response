import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load relevant datasets to be merged into pandas DataFrame (Extract)

    Args:
        messages_filepath (str): Filepath for disaster messages data
        categories_filepath (str): Filepath for disaster categories data

    Returns:
        df: A merged pandas DataFrame of the two datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    '''
    Clean the dataframe for machine learning model (Transform)

    Args:
        df: Raw pandas DataFrame

    Returns:
        df: Clean pandas DataFrame
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Store a clean pandas DataFrame to sqlite database (Load)

    Args:
        df: Clean pandas DataFrame
        database_filename (str): Name of the database
        
    Returns:
        None
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('disaster_cleaned', engine, index = False, if_exists = 'replace') 
    return 

def main():
    '''
    Main data function for ETL Pipeline
    
    1) Call load_data() -> extract
    2) Call clean_data() -> transform
    3) Call save_data() -> load

    Args:
        None
        
    Returns:
        None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()