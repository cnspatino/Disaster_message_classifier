import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    This function loads messages and categories datasets and merges them into one dataframe.

    Inputs: messages_filepath (string), categories_filepath (string)
    Output: df (pandas dataframe)

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):
    """
    This function processes the dataframe for machine learning and returns a cleaned dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories
    category_colnames = row.str[:-2].values.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

     # keep the last character of each string in categories (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    This function saves the dataframe to a specified database file.

    Inputs: df (pandas dataframe), database_filename (string)
    Outputs: None
    """
    # create and connect to database engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # save as sql table
    df.to_sql('MessageClassification', engine, if_exists = 'fail', index=False)


def main():
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