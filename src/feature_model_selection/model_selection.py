# Author: Fanli Zhou
# Date: 2021-02-15

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# classifiers / models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from lightgbm import LGBMClassifier

# other
from sklearn.metrics import roc_auc_score, auc, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from yellowbrick.classifier import ROCAUC
from sklearn.decomposition import PCA

sys.path.append('..')
from feature_model_selection.feature_selection import fit_and_report, get_scaled

def get_split_data(df, SEED):
    """
    Split data.
    Parameters:
    -----------
    df: pd.DataFrame
        feature data
    SEED: int
        random seed

    Returns:
    --------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        X_train_valid, X_train, X_valid, X_test, y_train_valid, y_train, y_valid, y_test
    """
    X = df.drop(columns=['tag']).to_numpy()
    y = df['tag'].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.20, random_state=SEED, stratify=y_train_valid)
    return X_train_valid, X_train, X_valid, X_test, y_train_valid, y_train, y_valid, y_test

def test_model(
    model_name, X_train, y_train, X_valid, y_valid,
    X_train_valid, y_train_valid, results, name,
    models, params, SEED, model_param={}, param={}):
    """
    Train a model and report the results and update the results dictionary.
    Parameters:
    -----------
    model_name: str
        name of the input model
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    X_valid: np.ndarray
        X for validation
    y_valid: np.ndarray
        y for validation
    X_train_valid: np.ndarray
        X for training and validation
    y_train_valid: np.ndarray
        y for training and validation
    results: dict
        results dictionary
    name: str
        model name
    models: dict
        model list
    params: dict
        parameters of the model
    SEED: int
        random seed
    model_param: dict (default: {})
        additional parameters
    param: dict (default: {})
        parameter dictionary
       
    Returns:
    --------
    list
        aucuracy scores
    """    
    auc = fit_and_report(
        models[model_name](**params[model_name], random_state=SEED, **model_param),
        X_train, y_train, X_valid, y_valid, param=param)
    auc.append(
        cross_val_score(
            models[model_name](**params[model_name], random_state=SEED, **model_param),
            X_train_valid, y_train_valid).mean()
    )

    update_results(results, auc, name)
    return auc

def oversampling(X, y, sampling_funcs, SEED, sampling_name=None):
    """
    Oversample the input X and y.
    Parameters:
    -----------
    model: Scikit-learn model
        input model
    X: np.ndarray
        feature values
    y: np.ndarray
        target
    sampling_funcs: dict
        sampling functions
    SEED: int
        random seed
    sampling_name: str (default: None)
        the sampling function name

    Returns:
    --------
    list
        [oversampled X, oversampled y]
    """
    return sampling_funcs[sampling_name](random_state=SEED).fit_resample(X, y)
 
def cross_val_oversample(
    model_name, X, y, sampling_name, models, params, SEED,
    cv=5, model_param={}, param={}):
    """
    Apply cross validation on oversampled data.
    Parameters:
    -----------
    model_name: str
        model name
    X: np.ndarray
        feature values
    y: np.ndarray
        target
    sampling_name: str
        the sampling function name
    models: dict
        model list
    params: dict
        parameters of the model
    SEED: int
        random seed
    cv: int (default: 5)
        cross-validation folder
    model_param: dict (default: {})
        additional parameters
    param: dict (default: {})
        parameter dictionary

    Returns:
    --------
    np.ndarray
        accuracy scores
    """    
    skf = StratifiedKFold(n_splits=cv, random_state=SEED, shuffle=True)
    auc = []
    for train_index, test_index in skf.split(X, y):

        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        X_train, y_train = oversampling(X_train, y_train, sampling_name, SEED)
        model = models[model_name](**params[model_name], **model_param)
        model.fit(X_train, y_train)
        if 'multi_class' not in param.keys():
            auc.append(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))
        else:
            auc.append(roc_auc_score(
                y_valid, model.predict_proba(X_valid), **param))
            
    return np.array(auc)
    

def test_oversampling(
    model_name, X_train, y_train, X_valid, y_valid,
    sampling_name, models, params, SEED, cv=5, model_param={}, param={}):
    """
    Test the effects of oversampling.
    Parameters:
    -----------
    model_name: str
        model name
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    X_valid: np.ndarray
        X for validation
    y_valid: np.ndarray
        y for validation
    sampling_name: str
        the sampling function name
    models: dict
        model list
    params: dict
        parameters of the model
    SEED: int
        random seed
    cv: int (default: 5)
        cross-validation folder
    model_param: dict (default: {})
        additional parameters
    param: dict (default: {})
        parameter dictionary

    Returns:
    --------
    float, list
        cross-validation accuracy score, accuracy scores
    """    
    cv_auc_score = cross_val_oversample(
        model_name, 
        X_train, y_train, sampling_name, models, params, 
        SEED, cv=cv, param=param
    ).mean()    
    print(f'The mean cross-validation auc score with {sampling_name} is {cv_auc_score:.5f}')
    X_train, y_train = oversampling(X_train, y_train, sampling_name, SEED)
    auc = fit_and_report(
        models[model_name](**params[model_name], **model_param), 
        X_train, y_train, X_valid, y_valid, param=param
    )
    return cv_auc_score, auc

def gs_optimize(model, X_train, y_train, parameters):
    """
    Optimizes hyperparameters for a given model.  
    Parameters:
    -----------
    model: Scikit-learn model
        input model
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    parameters: dict
        parameters of the model

    Returns:
    --------
    GridSearchCV object
        grid search results
    """
    gs = GridSearchCV(estimator=model,
                      param_grid=parameters, 
                      scoring='roc_auc',
                      cv=5,
                      return_train_score=True)

    gs.fit(X_train, y_train)
    print("Best parameters : %s" % gs.best_params_)
    return gs

def update_results(results, auc, model):
    """
    Update model results in the results dictionary. 
    Parameters:
    -----------
    results: dict
        results dictionary
    auc: list
        accuracies
    model: str
        model name
    """
    results['model'].append(model)
    results['train auc'].append(auc[0])
    results['valid auc'].append(auc[1])
    results['cv auc'].append(auc[2])

def optimize_fit_get_score(X_train, y_train, classifiers, parameters, SEED):
    """ 
    Optimizes hyperparameters for each model first.
    Parameters:
    -----------
    X_train: np.ndarray
        X for training
    y_train: np.ndarray
        y for training
    classifiers: dict
        model dictionary
    parameters: dict
        parameters of the model
    SEED: int
        random seed

    Returns:
    --------
    dict
        best hyperparameters results 
    """
    results = {}
    
    for classifier_name, classifier_obj in classifiers.items():
        print("Fitting", classifier_name)
        classifier_obj = RandomizedSearchCV(estimator=classifier_obj,
                                            random_state=SEED,
                                            param_distributions=parameters[classifier_name], 
                                            scoring='roc_auc',
                                            cv=5)
        classifier_obj.fit(X_train, y_train)
        print("Best parameters : %s" % classifier_obj.best_params_)
        results[classifier_name] = classifier_obj.best_params_
    return results

def model_selection(
    df, models, params,
    sampling_funcs, SEED,
    pca=False, over_sampling=True, 
    parameters = {
    'RandomForestClassifier': {
        'n_estimators': [5, 10, 15, 20, 25],
        'max_depth': [5, 7, 9, 11, 12]},
    'LGBMClassifier'         : {
        'n_estimators': [15, 20, 25, 30, 35],
        'max_depth': [5, 7, 9, 11, 12]},
    'GaussianProcessClassifier': {
        'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]}
    }
):
    """
    Apply model selection.
    Parameters:
    -----------
    df: pd.DataFrame
        feature data
    models: dict
        model list
    params: dict
        parameters of the model
    sampling_funcs: dict
        sampling functions
    SEED: int
        random seed
    pca: bool (default: False)
        whether to apply PCA
    over_sampling: bool (default: True) 
        whether to apply over sampling
    parameters: dict (default: {
        'RandomForestClassifier': {
            'n_estimators': [5, 10, 15, 20, 25],
            'max_depth': [5, 7, 9, 11, 12]},
        'LGBMClassifier'         : {
            'n_estimators': [15, 20, 25, 30, 35],
            'max_depth': [5, 7, 9, 11, 12]},
        'GaussianProcessClassifier': {
            'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]}
    })
        parameters for models
    """
    X_train_valid, X_train, X_valid, X_test, \
    y_train_valid, y_train, y_valid, y_test = \
    get_split_data(df, SEED)
    
    if pca:
        pca = PCA(n_components=5)
        # Learning the components and transforming X into Z
        X_train = pca.fit_transform(X_train)
        X_valid = pca.transform(X_valid)
        pca = PCA(n_components=5)
        X_train_valid = pca.fit_transform(X_train_valid)
        X_test = pca.transform(X_test)
       

    X_train, X_valid = get_scaled(X_train, X_valid)
    X_train_valid, X_test = get_scaled(X_train_valid, X_test)
    
    results = {
        'model': [],
        'train auc': [],
        'valid auc': [],
        'cv auc': []}

    print(f'\nBaseline model:')
    test_model(
        'LogisticRegression', X_train, y_train, X_valid, 
        y_valid, X_train_valid, y_train_valid, results, 
        'base lr', models, params, SEED
    )
    
    print(f'\nLogisticRegression optimization:')
    print('optimizing parameter C:')
    parameter_C = {'C': [0.01, 0.1, 1, 10]}

    gs = gs_optimize(
        LogisticRegression(solver='liblinear', random_state=SEED),
        X_train, y_train, parameter_C)
    
    test_model(
        'LogisticRegression', X_train, y_train, X_valid, 
        y_valid, X_train_valid, y_train_valid, results,
        'lr + optC', models, params, SEED, model_param=gs.best_params_
    )
    
    if over_sampling:
        print(f'\nHandle the class imbalance problem')
        print('Tuning class_weight')
        if results['cv auc'][-1] > results['cv auc'][-2]:
            print(f'Using optimized C for LogisticRegression: C = {gs.best_params_["C"]}')
            lr_param = {'C': gs.best_params_['C']}
            lr_cw_param = {'class_weight': 'balanced', 'C': gs.best_params_['C']} 
        else:
            lr_param = {}
            lr_cw_param = {'class_weight': 'balanced'} 
        test_model(
            'LogisticRegression', X_train, y_train, X_valid, 
            y_valid, X_train_valid, y_train_valid, results, 
            'lr + cw', models, params, SEED, model_param=lr_cw_param
        )
        print(f'\nOversampling')
        sampling_method = ''
        max_auc = 0
        max_diff = 1
        for sampling_name in sampling_funcs.keys():
            print(f'{sampling_name} oversampling')
            cv_auc_score, auc = (test_oversampling(
                'LogisticRegression', 
                X_train, y_train, X_valid, y_valid, sampling_name,
                models, params, SEED
            ))
            if cv_auc_score >= max_auc and abs(auc[0] - auc[1]) < max_diff:
                max_auc = cv_auc_score
                max_diff = abs(auc[0] - auc[1])
                sampling_method = sampling_name
        print(f'Using the {sampling_method} method for oversampling')

        X_train_os, y_train_os = oversampling(
            X_train, y_train, sampling_method, SEED)
        X_train_valid_os, y_train_valid_os = oversampling(
            X_train_valid, y_train_valid, sampling_method, SEED)

        test_model(
            'LogisticRegression', X_train_os, y_train_os, X_valid, 
            y_valid, X_train_valid_os, y_train_valid_os, results, 
            'lr + os', models, params, SEED, model_param=lr_param
        )
    
    print(f'\nOther classifiers:')
    classifiers = {
    'RandomForestClassifier' : RandomForestClassifier(random_state=SEED),
    'LGBMClassifier'          : LGBMClassifier(random_state=SEED),
    'GaussianProcessClassifier': GaussianProcessClassifier(random_state=SEED),
    }


    best_params = optimize_fit_get_score(X_train, y_train, classifiers, parameters, SEED)
    
    print(f'\nRandomForestClassifier:')
    test_model(
        'RandomForestClassifier', X_train, y_train, X_valid, 
        y_valid, X_train_valid, y_train_valid, results, 
        'rf + opt', models, params, SEED, model_param=best_params['RandomForestClassifier']
    )
    
    print(f'\nLGBMClassifier:')
    test_model(
        'LGBMClassifier', X_train, y_train, X_valid, 
        y_valid, X_train_valid, y_train_valid, results, 
        'lgbm + opt', models, params, SEED, model_param=best_params['LGBMClassifier']
    )
    
    print(f'\nGaussianProcessClassifier:')
    test_model(
        'GaussianProcessClassifier', X_train, y_train, X_valid, 
        y_valid, X_train_valid, y_train_valid, results, 
        'gpc + opt', models, params, SEED, model_param=best_params['GaussianProcessClassifier']
    )
    if over_sampling:
        return pd.DataFrame(results), lr_param, lr_cw_param, best_params, X_train_valid, y_train_valid, X_train_valid_os, y_train_valid_os, X_test, y_test
    else:
        return pd.DataFrame(results), gs.best_params_, best_params, X_train_valid, y_train_valid, X_test, y_test

def test_best_model(path_to_root, model, X_train_valid, y_train_valid, X_test, y_test, name):
    """
    Test and visualize the best model on test.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    model: Scikit-learn model
        input model
    X_train_valid: np.ndarray
        X for training and validation
    y_train_valid: np.ndarray
        y for training and validation
    X_test: np.ndarray
        X for testing
    y_test: np.ndarray
        y for testing
    sampling_name: str
        the sampling function name
    models: dict
        model list
    name: str
        task name
    """
    model.fit(X_train_valid, y_train_valid)
    pred = model.predict_proba(X_test)[:, 1]

    pd.DataFrame(
        {'target': y_test,
         'pred': pred}
    ).to_csv(f'{path_to_root}/results/{name}_results.csv', index=False)
    
    fpr, tpr, thresholds = roc_curve(y_test, pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    
    pd.DataFrame(
        {'x': fpr,
         'y': tpr,
         'tag': f'M{name[1:]} (AUC = {roc_auc: .2f})'
        }
    ).to_csv(f'{path_to_root}/results/{name}_plot_roc.csv', index=False)
    
    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        estimator_name='example estimator')
    display.plot()
    plt.show()
    
    visualizer = ROCAUC(model, classes=[0, 1])

    visualizer.fit(X_train_valid, y_train_valid)        
    visualizer.score(X_test, y_test)
    visualizer.show()