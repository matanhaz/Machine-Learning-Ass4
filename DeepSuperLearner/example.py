import os
import warnings
from collections import OrderedDict

from deepSuperLearnerLib import DeepSuperLearner

warnings.filterwarnings('ignore')
from sklearn.ensemble._forest import ExtraTreesClassifier as ExtremeRandomizedTrees
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._forest import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from numpy.random import *
from sklearn import datasets
#from deepSuperLearnerLib import *
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
import xlsxwriter
from sklearn import metrics
import time as timer
import pandas as pd
import itertools
import tqdm
import numpy

super_learner_features = "ERT: n_estimatores=200, max_features=1," \
                         "knn: n_neighbores=11," \
                         "rfc: b_estimatores=200," \
                         "xgb: n_estimatores=200, max_depth=3, lr=1"


def generate_classification():
    X, y = datasets.make_classification(n_samples=np.random.randint(800, 1500), n_features=np.random.randint(10, 23),
                                        n_informative=2, n_redundant=6)
    return X, y


def get_classification(arr):
    ret_arr = []
    for i in range(arr.shape[0]):
        maximum = max(arr[i])
        for j in range(arr[i].shape[0]):
            if arr[i][j] == maximum:
                ret_arr.append(j)
                continue
    return np.array(ret_arr)


def get_scores(model_name, iteration_number, best_model_and_parameters, data_set_name):
    size = best_model_and_parameters['X test'].size
    model = best_model_and_parameters['best model']
    start_time = timer.time()
    y_pred = model.predict(best_model_and_parameters['X test'])
    time_passed_inference = ((timer.time() - start_time) / size) * 1000
    y_true = best_model_and_parameters['Y test']
    if model_name != 'knn':
        y_pred = get_classification(y_pred)

    cm = metrics.confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    accuracy_old = (tp + tn) / (tp + fp + tn + fn)

   # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)

    accuracy_func = accuracy_score(best_model_and_parameters['Y test'], y_pred)
    accuracy_new = (tp + tn) / (tp + fp + tn + fn)
    precision = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)['macro avg']['precision']

    #auc_roc = roc_auc_score(y_true, model.predict(best_model_and_parameters['X test'])[:, 1])
    # auc_pr = metrics.auc(FPR,TPR)
    # a = model.predict(best_model_and_parameters['X test'])
    # b = model.predict(best_model_and_parameters['X test'],True)

    if model_name == 'knn':
        pr_curve = precision_recall_curve(y_true, model.predict_proba(best_model_and_parameters['X test'])[:, 1])
        auc_roc = roc_auc_score(y_true, model.predict(best_model_and_parameters['X test']))
    else:
        pr_curve = precision_recall_curve(y_true, model.predict(best_model_and_parameters['X test'])[:, 1])
        auc_roc = roc_auc_score(y_true, model.predict(best_model_and_parameters['X test'])[:, 1])

    auc_pr = metrics.auc(pr_curve[1], pr_curve[0])
    scores = [data_set_name, model_name, iteration_number, super_learner_features if model_name != 'knn' else "knn: n_neighbores=11" ,
              np.mean(accuracy_old), np.mean(TPR), np.mean(FPR), precision, auc_roc, auc_pr, best_model_and_parameters['train time'],
              time_passed_inference]


    return scores


def write_scores(model_to_best, data_set_name = None):
    if data_set_name is None:
        data_set_name = ""
    workbook = xlsxwriter.Workbook(f'results_{data_set_name}.xlsx')
    worksheet = workbook.add_worksheet()

    initial_data = ['Dataset Name', 'Algorythm Name', 'Cross Validation[1-10]', 'Hyper-Parameters Values',
                    'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inferences Time']

    for i in range(12):
        worksheet.write(0, i, initial_data[i])

    row = 1
    for data_set_name in model_to_best.keys():
        for model in model_to_best[data_set_name].keys():
            for iteration in model_to_best[data_set_name][model].keys():
                col = 0
                scores = get_scores(model, iteration, model_to_best[data_set_name][model][iteration], data_set_name)
                for score in scores:
                    worksheet.write(row, col, score)
                    col += 1
                row += 1
    workbook.close()


def cross_validation(X, Y, models, space):
    print("cross validation:")
    model_to_iteration = {}
    for _, model_name in models:
        model_to_iteration[model_name] = {}

    final_best_model = None

    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    index = 1
    index_to_best = {}
    # enumerate splits
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = Y[train_ix], Y[test_ix]

        # smote = SMOTE( )
        # X_train,y_train = smote.fit_resample( X_train, y_train )

        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        #search = RandomizedSearchCV(model, space, cv=cv_inner, refit=True, scoring='f1', n_iter=1)
        for model,model_name in models:
            start_time = timer.time()
            result = model.fit(X_train, y_train)
            #result = search.fit(X_train, y_train)
            time_passed_train = timer.time() - start_time

            # get the best performing model fit on the whole training set
            best_model = model
            best_params = []
            print("finished 1 cross fold validation!")

            # evaluate model on the hold out dataset
            model_to_iteration[model_name][index] = {
                "best model": best_model,
                "best params": best_params,
                "X test": X_test,
                "Y test": y_test,
                "train time": time_passed_train
            }
        index += 1

    return model_to_iteration

all_datasets = {"abalon.csv", "acute-inflammation.csv", "acute-nephritis.csv",
                "annealing.csv", "ar4.csv", "bank.csv", "blood.csv", "bodyfat.csv",
                "breast-cancer.csv", "breast-cancer-wisc.csv", "breast-cancer-wisc-diag.csv",
                "breast-cancer-wisc-prog.csv", "breast-tissue.csv", "car.csv", "chatfield_4.csv",
                "chscase_vine1.csv", "cloud.csv", "congressional-voting.csv",
                "conn-bench-sonar-mines-rocks.csv", "conn-bench-vowel-deterding.csv"}

def dataset_builder(dataset_name):
    X = None
    y = None
  
    path_to_data = r'classification_datasets' + '\\' + dataset_name

    if dataset_name == "abalon.csv":
        df = pd.read_csv(path_to_data)
        y = df['class']
        y = np.array(y)
        X = df.drop("class", axis=1)
        X = np.array(X)
    elif dataset_name == "ar4.csv":
        df = pd.read_csv(path_to_data)
        y = df['label']
        y = np.array(y)
        X = df.drop("label", axis=1)
        X = np.array(X)
    elif dataset_name == "bodyfat.csv" or dataset_name == "chatfield_4.csv" or dataset_name == "chscase_vine1.csv":
        df = pd.read_csv(path_to_data)
        df = df.replace("P", 1)
        df = df.replace("N", 0)
        y = df['binaryClass']
        y = np.array(y)
        X = df.drop("binaryClass", axis=1)
        X = np.array(X)
    elif dataset_name == "cloud.csv":
        df = pd.read_csv(path_to_data)
        df = df.replace("P", 1)
        df = df.replace("N", 0)
        df = df.replace("seeded", 1)
        df = df.replace("unseeded", 0)
        y = df['binaryClass']
        y = np.array(y)
        X = df.drop("binaryClass", axis=1)
        X = np.array(X)
    elif dataset_name in all_datasets:
        df = pd.read_csv(path_to_data)
        y = df['clase']
        y = np.array(y)
        X = df.drop("clase", axis=1)
        X = np.array(X)
    else:
        raise Exception("wrong dataset name")

    return (X, y)

import sys
if __name__ == '__main__':
    dic = {}
    got_argument = False
    if len(sys.argv) == 2:
        got_argument = True
        all_datasets = [str(sys.argv[1])]
    for dt_name in all_datasets:
        data_Set_name = dt_name
        #data_Set_name = "blood.csv"
        print("starting dsl")
        model_to_iteration_and_best = {}
        '''creating the learners for the super learner'''
        ERT_learner = ExtremeRandomizedTrees(n_estimators=200, max_depth=None, max_features=1)
        kNN_learner = kNearestNeighbors(n_neighbors=11)
        LR_learner = LogisticRegression()
        RFC_learner = RandomForestClassifier(n_estimators=200, max_depth=None)
        XGB_learner = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=1., verbosity=0)
        Base_learners = {'ExtremeRandomizedTrees': ERT_learner, 'kNearestNeighbors': kNN_learner,
                         'LogisticRegression': LR_learner,
                         'RandomForestClassifier': RFC_learner, 'XGBClassifier': XGB_learner}
        DSL_learner = DeepSuperLearner(Base_learners)
        # np.random.seed(100)

        '''generating data base , may be swapped to an exiting data set who knows'''
        X, y = dataset_builder(data_Set_name)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        '''fitting the model'''
        # DSL_learner.fit(X_train, y_train,max_iterations=20,sample_weight=None)
        space = {
            'n_neighbors': [11],
            'Trees': [200],
            'Depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max features when splitting': [1],
            'Max Depth': [3],
            'Row subsampling': [0],
            'Column subsampling': [0],
            'Learning rate': [1]
        }
        space = {}
        # DSL_learner.fit(X,y)
        # print()
        #model_to_iteration_and_best["DSL"] = cross_validation(X, y, DSL_learner, space)

        # DSL_learner.get_precision_recall(X_test, y_test, show_graphs=True)
        # print()

        '''now testing the improved model'''

        print("starting dsl improved")
        Base_learners_improved1 = {'ExtremeRandomizedTrees1': ERT_learner, 'ExtremeRandomizedTrees2': ERT_learner,
                                  'kNearestNeighbors1': kNN_learner, 'kNearestNeighbors2': kNN_learner,
                                  'LogisticRegression1': LR_learner, 'LogisticRegression2': LR_learner,
                                  'RandomForestClassifier1': RFC_learner, 'RandomForestClassifier2': RFC_learner,
                                  'XGBClassifier1': XGB_learner, 'XGBClassifier2': XGB_learner}

        DSL_learner_improved1 = DeepSuperLearner(Base_learners_improved1)
        #model_to_iteration_and_best["DSL_Improve1"] = cross_validation(X, y, DSL_learner_improved1, space)

        algos_names = ['ExtremeRandomizedTrees2', 'kNearestNeighbors2', 'LogisticRegression2',  'RandomForestClassifier2', 'XGBClassifier2']

        algos_names = numpy.random.permutation(algos_names)
        #algos_names = itertools.permutations(algos_names)

        algos_dict = OrderedDict()
        for n in algos_names:
            algos_dict[n] = Base_learners_improved1[n]

        Base_learners_improved2 = OrderedDict()
        Base_learners_improved2['ExtremeRandomizedTrees1'] = ERT_learner
        Base_learners_improved2['kNearestNeighbors1'] = kNN_learner
        Base_learners_improved2['LogisticRegression1'] = LR_learner
        Base_learners_improved2['RandomForestClassifier1'] = RFC_learner
        Base_learners_improved2['XGBClassifier1'] = XGB_learner

        for e in algos_dict:
            Base_learners_improved2[e] = algos_dict[e]

        DSL_learner_improved2 = DeepSuperLearner(Base_learners_improved2)
        #model_to_iteration_and_best["DSL_Improve2"] = cross_validation(X, y, DSL_learner_improved2, space)

        # Base_learners_improved2 = {'ExtremeRandomizedTrees1': ERT_learner,
        #                           'kNearestNeighbors1': kNN_learner,
        #                           'LogisticRegression1': LR_learner,
        #                           'RandomForestClassifier1': RFC_learner,
        #                           'XGBClassifier1': XGB_learner}

        space = {
            'Neighbors': [11],
            'Trees': [200],
            'Depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max features when splitting': [1],
            'Max Depth': [3],
            'Row subsampling': [0],
            'Column subsampling': [0],
            'Learning rate': [1]
        }
        space = {}
        # model_to_iteration_and_best["DSL improved"] = cross_validation(X, y, DSL_learner, space)

        '''now testing a third algorithm'''
        print("starting knn")
        kNN_learner = kNearestNeighbors(n_neighbors=11)
        space = {
            'Neighbors': [11],
            'Trees': [200],
            'Depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max features when splitting': [1],
            'Max Depth': [3],
            'Row subsampling': [0],
            'Column subsampling': [0],
            'Learning rate': [1]
        }

        space = {}

        #model_to_iteration_and_best["knn"] = cross_validation(X, y, kNN_learner, space)
        models = [(DSL_learner,"DSL"),(DSL_learner_improved1,"DSL improved 1"),(DSL_learner_improved2,"DSL improved 2"),(kNN_learner,"knn")]
        models_to_iterations_dict = cross_validation(X, y, models, space)

        dic[data_Set_name] = models_to_iterations_dict
        if got_argument:
            break
    write_scores(dic, data_Set_name)
