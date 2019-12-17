import os, pickle
from utils_data.data_prep import load_data_clean, make_calendar, data_preparation, creation_features, transform_data
from utils_data.train import train
from sklearn.ensemble import RandomForestClassifier
from utils_data.prediction import paria_prediction

import warnings
warnings.filterwarnings("ignore")

usecols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
urls = ['https://www.football-data.co.uk/mmz4281/{}{}/F1.csv'.format(i,i+1) for i in range(15, 20)]
MODEL = 'model/rf.pkl'

df, teams = load_data_clean(urls, usecols)

url_cal = '../data/calendrier_l1.csv'
calendar = make_calendar(url_cal, teams)

if os.path.exists(MODEL):
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

day = int(input('Enter the day of the prediction: \n'))
# prediction
print(paria_prediction(day, teams, df, calendar, model))
