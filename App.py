from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import random

app = Flask(__name__, static_url_path='/static')

# Load your dataset
teams = pd.read_csv("teams.csv")  # Update the path if needed
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
teams[['athletes', 'prev_medals']] = imputer.fit_transform(teams[['athletes', 'prev_medals']])

# Train a linear regression model
train = teams[teams["year"] < 2012].copy()
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        athletes = float(request.form['athletes'])
        prev_medals = float(request.form['prev_medals'])

        # Make predictions using the trained model
        prediction = reg.predict([[athletes, prev_medals]])

        # Introduce random noise to the prediction
        noise = random.uniform(-0.1, 0.1)  # You can adjust the range of the noise
        prediction = max(0, prediction[0] + noise)  # Ensure predictions are non-negative

        # Pass the prediction to the template
        return render_template('result.html', athletes=athletes, prev_medals=prev_medals, prediction=prediction)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

@app.route('/datasets')
def datasets():
    return render_template('datasets.html', teams=teams)

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
