import pandas as pd
from datetime import datetime
import difflib
import numpy as np

def load_data_clean(urls, usecols, date_format = '%d%m%y'):
    # load data
    df = pd.concat((pd.read_csv(url, usecols=usecols) for url in urls), ignore_index=True).dropna()
    # create the Date
    date_converter = lambda x: ''.join([x.replace('/20', '').replace('/', '')[i:i+2] for i in range(0, len(x.replace('/20', '').replace('/', '')), 2)])
    df['Date'] = pd.to_datetime(df.Date.apply(date_converter), format=date_format)
    df['HomeTeam'], team_unique = df.HomeTeam.factorize()
    df['AwayTeam'] = df.AwayTeam.apply(lambda x: list(team_unique).index(x))
    return df, list(team_unique)

def make_calendar(url, teams):
    calendar = pd.read_csv(url, index_col=None).drop(columns=['Unnamed: 0', 'score', 'url'])
    pat = '(AS|Olympique|SCO|Stade|FC|RC)'
    update_name = lambda x: difflib.get_close_matches(x, teams, cutoff=0.1, n=1)[0] if x != 'LOSC' else 'Lille'
    calendar.team1 = calendar.team1.str.replace(pat, '').apply(update_name).apply(lambda x: teams.index(x))
    calendar.team2 = calendar.team2.str.replace(pat, '').apply(update_name).apply(lambda x: teams.index(x))
    return calendar

def trfrm_dat(df, var):
    df[var] = np.where(df['Team'] == df['HomeTeam'], df['H{}'.format(var)], df['A{}'.format(var)])
    df['{}_Opp'.format(var)] = np.where(df['Team'] != df['HomeTeam'], df['H{}'.format(var)], df['A{}'.format(var)])
    return df

def data_preparation(df):
    # copy dataset to avoid to write on;
    df_copy = df.copy();

    # unpivot data
    df_copy['H'] = df_copy['HomeTeam']
    df_copy['A'] = df_copy['AwayTeam']
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    team_results = pd.melt(df_copy, id_vars=cols_to_keep, value_vars=['H', 'A'], var_name='Home/Away', value_name='Team')
    team_results['Opponent'] = np.where(team_results['Team'] == team_results['HomeTeam'], team_results['AwayTeam'], team_results['HomeTeam'])
    team_results['FTG'] = np.where(team_results['Team'] == team_results['HomeTeam'], team_results['FTHG'], team_results['FTAG'])
    team_results['FTG_Opp'] = np.where(team_results['Team'] != team_results['HomeTeam'], team_results['FTHG'], team_results['FTAG'])

    # stats for the team
    col = ['S', 'ST', 'F', 'C', 'Y', 'R']
    for var in col:
        trfrm_dat(team_results, var)

    # result W, D, L
    team_results['final_result'] = team_results.apply(lambda x: 'D' if x['FTR'] == 'D' else 'W' if x['Home/Away'] == x['FTR'] else 'L', axis=1)
    col_todrop = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    team_results.drop(columns=col_todrop, inplace=True)
    return team_results

def creation_features(df, k=5):
    # create new variables based on historical result
    col_to_shift = list(df.columns[4:])
    for sort_by in ['Team', 'Opponent']:
        df = df.sort_values(by=[sort_by, 'Date'])
        for var in col_to_shift:
            for i in range(1, k+1):
                df['{}_{}_{}'.format(var, sort_by, i)] = df.groupby([sort_by])[var].shift(i)

    # Time
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)

    # Result overall by Team and Year
    f = lambda x: tuple(sum([1 if x is r else 0 for x in list(x)])/len(x) for r in ['W', 'D', 'L'])
    to_m = df.groupby(['Team', 'Year']).final_result.apply(f).apply(pd.Series, index=["scoret_w", "scoret_d", "scoret_l"]).reset_index()
    df_3 = pd.merge(right=pd.merge(right=df, left=to_m, left_on=['Team', 'Year'], right_on=['Team', 'Year']), left=to_m, left_on=['Team', 'Year'], right_on=['Opponent', 'Year'])

    data = df_3[['Date', 'Home/Away', 'Team_y', 'Opponent', 'final_result', "scoret_w_x", "scoret_d_x", "scoret_l_x", "scoret_w_y", "scoret_d_y", "scoret_l_y"]+list(df.columns[19:])]

    return data

def transform_data(data):
    # Features + target
    data = data.dropna()
    X = data.drop(columns='final_result')
    y = data.final_result
    # transform
    X_obj = X.select_dtypes('object').apply(lambda x: x.factorize()[0], axis=0)
    X_con = X[['Year', 'Month']+list(X.select_dtypes('float').columns)]
    #pca = PCA(n_components=30, random_state=0)
    #X_con = pca.fit_transform(X_con)
    X = pd.concat([X_obj, X_con], axis=1)
    return X, y
