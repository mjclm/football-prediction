# for app
from flask import Flask, request, render_template, session, redirect
# for ML
import os, pickle, time
from utils_data.data_prep import load_data_clean, make_calendar, data_preparation, creation_features, transform_data
from utils_data.train import train
from sklearn.ensemble import RandomForestClassifier
from utils_data.prediction import paria_prediction

app = Flask(__name__)

# prediction
usecols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
urls = ['https://www.football-data.co.uk/mmz4281/{}{}/F1.csv'.format(i,i+1) for i in range(15, 20)]
MODEL = 'model/rf.pkl'
df, teams = load_data_clean(urls, usecols)
# calender of the league
url_cal = '../data/calendrier_l1.csv'
calendar = make_calendar(url_cal, teams)

# load or train the model
if os.path.exists(MODEL) and ((time.time() - os.path.getctime(MODEL)) > 6e5):
    model = pickle.load(open(MODEL, 'rb'))
else:
    X, y = transform_data(creation_features(data_preparation(df)))
    rf = RandomForestClassifier(class_weight='balanced',random_state=1)
    param_grid = {
        'model__n_estimators':[77, 80],
        'model__max_depth':[5, 7],
        'model__min_samples_leaf':[6, 7]
    }
    model = train(X, y, model=rf, param_grid=param_grid)
    pickle.dump(model, open(MODEL, 'wb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction")
def predict():
    return render_template("prediction.html")

@app.route('/prediction', methods=['POST', "GET"])
def predict_post():
    day = int(request.form['day'])
    df_pred = paria_prediction(day, teams, df, calendar, model).round(2).reset_index()
    return render_template("temp_df.html", data=df_pred.to_html(index=False, classes='table table-striped table-dark'))

if __name__ == "__main__":
    app.run(debug=True)
