# Author: Fanli Zhou
# Date: 2021-02-15

import numpy as np
import pandas as pd
from collections import Counter

# Feature selection
from sklearn.feature_selection import SequentialFeatureSelector, f_classif

# other
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def get_scaled(X_train, X_test):
    """
    Get scaled X_train, X_test.
    Parameters:
    -----------
    X_train: np.ndarray
        X for training
    X_test: np.ndarray
        X for testing

    Returns:
    --------
    np.ndarray, np.ndarray
        scaled X_train, scaled X_test
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def fit_and_report(
    model, X_train, y_train, X_test, y_test, param={}):
    """
    Fit model and report results.
    Parameters:
    -----------
    model: Scikit-learn model
        input model
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    X_test: np.ndarray
        X for testing
    y_test: np.ndarray
        y for testing
    param: dict (default: {})
        parameter dictionary

    Returns:
    --------
    list
        aucuracy scores
    """
    model.fit(X_train, y_train)
    if 'multi_class' not in param.keys():
        auc = [roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 
               roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])]   
    else:
        auc = [roc_auc_score(y_train, model.predict_proba(X_train), **param), 
               roc_auc_score(y_test, model.predict_proba(X_test), **param)]   
        
    print(f'Cross-validation auc: {auc[0]}\ntest auc: {auc[1]}')
    return auc

def get_best_params(
    X_train, y_train, model_name, param_grid, 
    models, base_params, cv=5, scoring='roc_auc'
):
    """
    Fit model and report results.
    Parameters:
    -----------
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    model_name: str
        name of the input model
    param_grid: dict
        parameter values to test
    models: dict
        model list
    base_params: dict
        parameters of the model
    cv: int (default: 5)
        cross-validation folder
    scoring: str (default: 'roc_auc')
        the scroing method

    Returns:
    --------
    dict
        best parameters for the model
    """
    model = GridSearchCV(
        models[model_name](**base_params[model_name]), 
        param_grid, cv=cv, scoring=scoring)
    model.fit(X_train, y_train);
    
    return model.best_params_

def fs_stat(X, y, columns):
    """
    Feature selection with ANOVA test.
    Parameters:
    -----------
    X: np.ndarray
        feature values
    y: np.ndarray
        target
    columns: list
        feature names
       
    Returns:
    --------
    list
        feature rank
    """
    feature_stat = pd.DataFrame({
        'features' : columns,
        'p-val': f_classif(X, y)[1]
    }).sort_values(by = 'p-val', ascending=True).reset_index(drop = True)

    return feature_stat

def fs_sfs(
    X, y, model_name, best_params, columns, 
    models, base_params, direction='forward', 
    scoring='roc_auc', cv=5
):
    """
    Feature selection with wrappers (forward or backward feature selection).
    Parameters:
    -----------
    X: np.ndarray
        feature values
    y: np.ndarray
        target
    model_name: str
        name of the input model
    best_params: dict
        best parameters for the model
    columns: list
        feature names
    models: dict
        model list
    base_params: dict
        parameters of the model
    direction: str (default: 'forward')
        wrapper type 'forward' or 'backward'
    scoring: str (default: 'roc_auc')
        the scroing method
    cv: int (default: 5)
        cross-validation folder
       
    Returns:
    --------
    list
        feature rank
    """
    sfs = SequentialFeatureSelector(
        models[model_name](**base_params[model_name], **best_params),
        n_features_to_select=5, direction=direction, scoring=scoring, cv=cv
    ).fit(X, y)
    
    return columns[sfs.get_support()]

def save_train_data(path_to_root, name, SEED):
    """
    Save data from training in each task.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    name: str
        task name
    SEED: int
        random seed
    """
    df = pd.read_csv(f'{path_to_root}/results/{name}_features.csv').iloc[:, 1:]

    if name == 'm1':
        df.tag.replace({1 : 0, 2 : 1}, inplace=True)

    elif name == 'm2':
        df = df[df.tag != 0]
        df.tag.replace({1 : 0, 2 : 1}, inplace=True)
        
    elif name == 'm3':
        df.tag.replace({2 : 1}, inplace=True)
        
    X = df.drop(columns = ['tag']).to_numpy()
    y = df['tag'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify = y)
    
    train_data = pd.DataFrame(
        np.concatenate((X_train, y_train.reshape(len(y_train), 1)), axis=1),
        columns=['l' + str(i) for i in range(X_train.shape[1])] + ['tag']
    )
    train_data.to_csv(f'{path_to_root}/results/{name}_features_train_data.csv', index=False)

def feature_selection(path_to_root, df, name, SEED, models, base_params):
    """
    Embedded feature selection.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    df: pd.DataFrame
        feature data
    name: str
        task name
    SEED: int
        random seed
    models: dict
        model list
    base_params: dict
        parameters of the model
    """
    X = df.drop(columns=['tag']).to_numpy()
    y = df['tag'].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.20, random_state=SEED, stratify=y_train_valid)

    X_train_valid, X_test = get_scaled(X_train_valid, X_test)
    X_train, X_valid = get_scaled(X_train, X_valid)
    columns = df.columns[:-1]
    
    # univariate feature selection
    feature_stat = fs_stat(X_train_valid, y_train_valid, columns)
    
    param_grid = {'C': [0.001, 0.01, 0.1]}
    best_params = get_best_params(
        X_train_valid, y_train_valid, 
        'LogisticRegression', param_grid, 
        models, base_params
    )
    
    # forward selection
    feature_forward = fs_sfs(
        X_train_valid, y_train_valid, 'LogisticRegression', best_params, 
        columns, models, base_params, direction='forward'
    )
    
    # backward selection
    feature_backward = fs_sfs(
        X_train_valid, y_train_valid, 'LogisticRegression', best_params, 
        columns, models, base_params, direction='backward'
    )
    
    # AUC-based permutation Importance for random forests
    feature_rf = pd.read_csv(f'{path_to_root}/results/{name}_features_imp.csv')
    
    features = \
    list(feature_stat.features[:5]) + list(feature_forward) + \
    list(feature_backward) + list(feature_rf.features[:5])
    
    dic = Counter(features)
    print(dic)
    pd.DataFrame({
        'features': dic.keys(),
        'votes': dic.values()
    }).to_csv(f'{path_to_root}/results/{name}_feature_selection.csv', index=False)    
    col = []
    for key in dic.keys():
        if dic[key] > 1:
            col.append(key)
    print(col)
    df_af_selected = df[[*col, 'tag']]
    df_af_selected.to_csv(f'{path_to_root}/results/{name}_selected_features.csv', index=False)