import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageClassification', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # extract genre data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data for columns that include 'related'
    related = df[['aid_related','infrastructure_related','weather_related']].sum().reset_index()
    related.columns = ['category','count']
    related_categories = related.category.values.tolist()
    related_counts = related['count'].values.tolist()

    # extract weather-related data
    weather = df[['floods','storm','fire','earthquake','cold','other_weather']].sum().reset_index()
    weather.columns = ['category','count']
    weather_categories = weather.category.values.tolist()
    weather_counts = weather['count'].values.tolist()

    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )   
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=related_categories,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=weather_categories,
                    y=weather_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Weather-related Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Weather Category"
                }
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