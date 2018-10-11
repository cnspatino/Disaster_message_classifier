# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This code runs using Python versions 3.*. Additional python packages needed for this code are:

json 
plotly
pandas
nltk
flask
sklearn
sqlalchemy
sqlite3
pkl

## Project Motivation <a name="motivation"></a>
My motivation for this project was to demonstrate writing an ETL pipeline and a machine learning pipeline to build a model for an API that classifies disaster messages. The web app displays visualizations of the disaster message data and also includes a textbox input where an emergency worker can input a new message and get classification results.  

## File Descriptions <a name="files"></a>

The ‘app’ folder contains the python script and html for the web app.

The 'data' folder contains two csv files to use as training data for the classifier: one with disaster categories and one with the corresponding disaster messages. In addition, this folder contains the python script for processing the data. The user can input their own data files when running this script if he or she chooses. 

The ‘models’ folder contains the python script for training and saving a machine learning model to classify the disaster message data.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>
This project was part of Udacity’s Data Science Nanodegree program. I'd like to thank Udacity for providing some template code for this project, especially for the web app.
