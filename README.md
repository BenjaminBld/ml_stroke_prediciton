# Stroke Prediction

This project applies a machine learning ensemble model to predict the likelihood of stroke occurrences based on various health metrics. The dataset includes a range of health indicators and stroke occurrences for numerous patients.

## Directory Structure

The project has the following structure:

- Download the dataset from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset and put it in `data/`
- `healthcare-dataset-stroke-data.csv`: The dataset containing health metrics and stroke occurrence for several patients.
- `src/`: Contains the source code for the project.
- `stroke_prediction_classifier.py`: The Python script that loads the data, trains the stroke prediction model, and evaluates its performance.
- `predictions/`: This directory contains the model's predictions on the test data.


## Running the Project

To run the project, navigate to the 'src/' directory in your terminal or command prompt and execute the 'stroke_prediction_classifier.py' script. This will load the data, train the models, evaluate their performance, and save the predictions.

## Installation

This project requires Python 3.6+ and several Python libraries which are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
