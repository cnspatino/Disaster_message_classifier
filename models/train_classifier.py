import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """
    This function loads data by reading a sql table from the specified database
    and then splits the data into features and target.

    Input: database_filepath (string)
    Outputs: X (features array), Y (target variables array), categories (list of category names)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessageClassification', engine)

    # designate features array and target array
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1).values

    # get list of category names
    categories = df.drop(['id','message','original','genre'], axis=1).columns.values.tolist()

    return X, Y, categories


def tokenize(text):
    """
    This function normalizes, lemmatizes, and tokenizes text.
    """
    # replace non-letter and non-number characters with a space, convert all to lowercase
    text = re.sub(r'[^a-zA-z0-9]', ' ', text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize the tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    This function builds a model using a machine learning pipeline and returns
    a gridsearch object ready to be trained on the training set.

    Input: None
    Output: cv (grid search object)
    """
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfid', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [5,10,20],
    }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the given model by predicting Y values from the test set
    and prints out the accuracy score and classification report for each Y column.
    """
    # predict Y values given X_test using model
    Y_preds = model.predict(X_test)

    # convert Y arrays to dataframes
    Y_preds = pd.DataFrame(Y_preds, columns = category_names)
    Y_test = pd.DataFrame(Y_test, columns = category_names)

    for column in Y_preds:
        accuracy = accuracy_score(Y_test[column], Y_preds[column])
        report = classification_report(Y_test[column], Y_preds[column])
        print(column+':')
        print(report)
        print('Accuracy: {:.4f}'.format(accuracy),'\n')


def save_model(model, model_filepath):
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
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