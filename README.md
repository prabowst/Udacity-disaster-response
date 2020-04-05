# Udacity Disaster Response

This project includes the deliverables for Udacity Data Science Nanodegree Program: Disaster Response Pipeline Project.

![front](https://github.com/prabowst/Udacity-disaster-response/blob/master/figures/disaster_response_front.PNG)

### Table of Contents 

1. [Requirements](#requirements)
2. [Project Description](#description)
3. [Files](#files)
4. [Run](#run)
5. [Project Results](#results)
6. [Licensing and Acknowledgements](#licensing)

### Requirements<a name="requirements"></a>

The code runs using Python version 3. Below are the list of packages used within this project.

1. pandas
2. numpy 
3. sqlalchemy
4. pickle
5. nltk
6. sklearn
7. json
8. plotly
9. joblib
10. flask

### Project Description<a name="description"></a>

The project aims to process disaster responses' messages to see what sort of action needs to be taken in addressing the problem. There are three different steps to this project:

1. ETL Pipeline
A data cleaning process is executed by loading `messages` and `categories` datasets. The datasets are merged, cleaned, and stored in a SQLite database. This process is captured in `process_data.py` Python script.

2. ML Pipeline
The machine learning pipeline trains and tunes machine learning model using GridSearchCV. The model is exported as a pickle file. This step is included in Python script, `train_classifier.py`.

3. Flask Web App
The web app is provided by Udacity with little changes done within the scope of this project. The names of database, model, and the graph to be represented on the web app are modified and added. 

### Files<a name="files"></a>

The files are organized as follows. In addition to what is required by Udacity, a `figures` folder is added containing several screenshots of the web app results.

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- figures
|- disaster_response_query.png # screenshot of front page web app
|- disaster_response_front.png # screenshot of query results

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### Run<a name="run"></a>

Instructions on how to run the program:

1. Run the following command in the project's root directory to set up your database.
ETL Pipeline: 
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
2. Run the following command in the project's root directory to set up your model.
ML Pipeline: 
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
3. Run the following command in the project's root directory to run your web app.
Web App: 
```
python app/run.py
```
Go to http://0.0.0.0:3001/ OR http://localhost:3001 to open the web app.

### Project Results<a name="results"></a>

The project results are shown by the screenshot below:

![query](https://github.com/prabowst/Udacity-disaster-response/blob/master/figures/disaster_response_query.PNG)

### Licensing and Acknowledgements<a name="licensing"></a>

Credit to Udacity course for the project ideas, code templates and data. The data sourced by Udacity are obtained from Figure Eight. This project web app was completed in part of Udacity Nanodegree program.
