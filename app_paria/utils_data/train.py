from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def train(X, y, model=None, param_grid=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe = Pipeline([('model', model)])
    search = GridSearchCV(pipe, param_grid, iid=True, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    return search
