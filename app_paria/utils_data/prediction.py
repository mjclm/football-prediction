from datetime import datetime
import pandas as pd
import numpy as np
from utils_data.data_prep import transform_data, creation_features, data_preparation

def paria_prediction(day, teams, df, calendar, model=None):
    cal = calendar.iloc[np.where(calendar.day == day)[0]]
    date = datetime.today()
    match_to_predict = {'Date': [date for _ in range(len(cal))], 'HomeTeam': cal.team1, 'AwayTeam': cal.team2, 'FTHG': np.NaN, 'FTAG': np.NaN,
         'FTR': np.NaN, 'HTHG': np.NaN, 'HTAG': np.NaN, 'HTR': np.NaN, 'HS': np.NaN, 'AS': np.NaN,
         'HST': np.NaN, 'AST': np.NaN, 'HF': np.NaN, 'AF': np.NaN, 'HC': np.NaN, 'AC': np.NaN,
         'HY': np.NaN, 'AY': np.NaN, 'HR': np.NaN, 'AR': np.NaN}
    row_pred = pd.DataFrame(match_to_predict)
    pred_data = creation_features(data_preparation(pd.concat([df, row_pred])))
    x = pred_data.sort_values('Date')[-20:]
    X2pred, _ = transform_data(x)
    prediction = model.predict_proba(X2pred)
    # create dataframe for prediction
    df_pred = pd.DataFrame(prediction, columns=['Drawback', 'Lose', 'Win'])
    df_pred[['Team', 'Opponent']] = x[['Team_y','Opponent']].reset_index(drop=True).applymap(lambda x: teams[x])
    df_pred['Home/Away'] = x['Home/Away'].reset_index(drop=True)
    df_pred['index_match'] = df_pred[['Team', 'Opponent']].apply(lambda x: str(sorted(x)), axis=1).factorize()[0]
    df_pred['Home'] = df_pred.apply(lambda x: x.Win if x['Home/Away'] is 'H' else x.Lose, axis=1)
    df_pred['Away'] = df_pred.apply(lambda x: x.Win if x['Home/Away'] is 'A' else x.Lose, axis=1)
    df_pred['HomeTeam'] = df_pred.apply(lambda x: x.Team if x['Home/Away'] is 'H' else x.Opponent, axis=1)
    df_pred['AwayTeam'] = df_pred.apply(lambda x: x.Team if x['Home/Away'] is 'A' else x.Opponent, axis=1)
    df4predict = df_pred.groupby(['HomeTeam', 'AwayTeam'])[['Home', 'Drawback', 'Away']].mean()
    return df4predict
