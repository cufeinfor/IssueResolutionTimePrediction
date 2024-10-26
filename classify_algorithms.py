from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from numpy import *
from training_test_sliding_validation import getData_slidingValidation_list, getData_all
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
import xlsxwriter as xw
import json

# K Nearest Neighbor
def selectKNNParam(data_path):
    n_neighbors = [i for i in range(1, 13)]
    # Selected parameters
    n_neighbors = [9]

    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for neighbors_num in n_neighbors:
        model = neighbors.KNeighborsClassifier(n_neighbors=neighbors_num)
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean, 'average macro f1_mean', macro_f1_mean,
              'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)

        return weight_f1_mean, accuracy_mean


# Support Vector Machine
def selectSVMParam(data_path):
    c_list = [0.1, 0.5, 1, 5, 10]
    # Selected parameters
    c_list = [0.1]

    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for c in c_list:
        model = svm.SVC(kernel='rbf',probability=True,C=c)
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean, 'average macro f1_mean', macro_f1_mean,
              'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)

        return weight_f1_mean, accuracy_mean

# Logistic Regression
def selectLRParam(data_path):
    c_list = [0.1,0.5,1.0,5, 10]
    # Selected parameters
    c_list = [0.1]
    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for c in c_list:
        model = LogisticRegression(C=c, max_iter=400)
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean, 'average macro f1_mean', macro_f1_mean,
              'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)

        return weight_f1_mean, accuracy_mean



# GaussianNB
def selectNBCParam(data_path):
    var_smoothing = [1,0.2,0.1,0.05,0.01]
    # Selected parameters
    var_smoothing = [1]
    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36, 4)
    for smooth in var_smoothing:
        model = GaussianNB(var_smoothing=smooth)
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean,'average macro f1_mean', macro_f1_mean, 'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
        return weight_f1_mean,accuracy_mean


# DTC
def selectDTCParam(data_path):
    max_depth = [2,3,4,5,6,7,8]
    # Selected parameters
    max_depth = [4]

    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36, 4)
    for depth in max_depth:
        model = tree.DecisionTreeClassifier(max_depth=depth, criterion='gini', )
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean,'average macro f1_mean', macro_f1_mean, 'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
        return weight_f1_mean,accuracy_mean

# MLP
def selectMLPParam(data_path):
    hidden_layer_sizes = [(100,),(2*21+1,),(int(math.log(21,2)),),(10,10),(20,20)]
    # Selected parameters
    hidden_layer_sizes = [(100,)]

    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for hidden_size in hidden_layer_sizes:
        model = MLPClassifier(hidden_layer_sizes=hidden_size,max_iter=400)
        accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
            model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean', macro_precision_mean,
              'average micro precision_mean', micro_precision_mean,
              'average weight precision_mean', weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )
        print('average weight f1_mean', weight_f1_mean, 'average macro f1_mean', macro_f1_mean,
              'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
        return weight_f1_mean, accuracy_mean

# GBDT
def selectGBDTParam(data_path):
    max_depth = [4,5,6,7,8,9,]
    lr_list = [0.1]
    # Selected parameters
    max_depth = [6]
    lr_list = [0.1]

    train_dataset = getData_all(data_path)
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for depth in max_depth:
        for lr in lr_list:
            model = GradientBoostingClassifier(max_depth=depth, learning_rate=lr, n_estimators=400)
            accuracy_mean, micro_precision_mean, micro_recall_mean, macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(
                model, x_train_list, y_train_list, x_validate_list, y_validate_list)
            print('average macro precision_mean', macro_precision_mean,
                  'average micro precision_mean', micro_precision_mean,
                  'average weight precision_mean', weight_precision_mean)
            print('average macro recall_mean', macro_recall_mean,
                  'average micro recall_mean', micro_recall_mean,
                  'average weight recall_mean', weight_recall_mean, )
            print('average weight f1_mean', weight_f1_mean,'average macro f1_mean', macro_f1_mean, 'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
            return weight_f1_mean,accuracy_mean

# RF
def selectRFParam(data_path):
    max_depth = [2, 3, 4, 5, 6, 7, 8]
    # Selected parameters
    max_depth = [6]

    train_dataset = getData_all(data_path)
    # # binary classification
    # min_class = np.min(train_dataset[:, -1])
    # for i, label in np.ndenumerate(train_dataset[:, -1]):
    #     num_add = 0
    #     if label > min_class + num_add:
    #         train_dataset[i, -1] = 0
    #     else:
    #         train_dataset[i, -1] = 1
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36,4)
    for depth in max_depth:
        model = RandomForestClassifier(max_depth=depth,n_estimators=400)
        # model = RandomForestClassifier(max_depth=5,n_estimators=1000)
        # Multiclassification
        accuracy_mean,  micro_precision_mean, micro_recall_mean,macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(model, x_train_list, y_train_list, x_validate_list, y_validate_list)
        print('average macro precision_mean',macro_precision_mean,
              'average micro precision_mean',micro_precision_mean,
              'average weight precision_mean',weight_precision_mean)
        print('average macro recall_mean', macro_recall_mean,
              'average micro recall_mean', micro_recall_mean,
              'average weight recall_mean', weight_recall_mean, )

        print('average weight f1_mean', weight_f1_mean,'average macro f1_mean', macro_f1_mean, 'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
        return weight_f1_mean,accuracy_mean
        # # Binary classification
        # precision_mean, recall_mean, accuracy_mean, f1_mean = get_model_train_by_slide_with_binary(model,
        #                                                                                            x_train_list,
        #                                                                                            y_train_list,
        #                                                                                            x_validate_list,
        #                                                                                            y_validate_list)
        # print('average precision_mean', precision_mean,
        #       'average recall_mean', recall_mean,
        #       'average accuracy_mean', accuracy_mean,
        #       'average f1_mean', f1_mean, )
        # return f1_mean, accuracy_mean
#XGBoost
def selectXGBoostParam(data_path):

    # n_estimators = [200, 400,500,600]
    max_depth = [4, 5, 6,7, 8,9,]
    learning_rate = [0.05, 0.1, 0.2]
    # Selected parameters
    max_depth = [6]
    learning_rate = [0.1]
    clf_XGBoost = XGBClassifier(n_estimators=400)

    train_dataset = getData_all(data_path)
    # # Binary classification
    # min_class = np.min(train_dataset[:,-1])
    # for i, label in np.ndenumerate(train_dataset[:,-1]):
    #     num_add = 0
    #     if label > min_class+num_add:
    #         train_dataset[i,-1] = 0
    #     else:
    #         train_dataset[i, -1] = 1
    x_train_list, y_train_list, x_validate_list, y_validate_list = getData_slidingValidation_list(train_dataset, 36, 4)
    for depth in max_depth:
        for lr in learning_rate:
            model = XGBClassifier(max_depth=depth, learning_rate=lr, n_estimators=400)
            # Multiclassification
            accuracy_mean,  micro_precision_mean, micro_recall_mean,macro_precision_mean, macro_recall_mean, weight_precision_mean, weight_recall_mean, weight_f1_mean, macro_f1_mean, micro_f1_mean, cohen_kappa_mean = get_model_train_by_slide(model, x_train_list, y_train_list, x_validate_list, y_validate_list)
            print('average macro precision_mean',macro_precision_mean,
                  'average micro precision_mean',micro_precision_mean,
                  'average weight precision_mean',weight_precision_mean)
            print('average macro recall_mean', macro_recall_mean,
                  'average micro recall_mean', micro_recall_mean,
                  'average weight recall_mean', weight_recall_mean, )

            print('average weight f1_mean', weight_f1_mean,'average macro f1_mean', macro_f1_mean, 'average micro f1_mean', micro_f1_mean, 'average accuracy_mean', accuracy_mean)
            return weight_f1_mean,accuracy_mean
            # # Binary classification
            # precision_mean, recall_mean, accuracy_mean, f1_mean = get_model_train_by_slide_with_binary(model, x_train_list, y_train_list, x_validate_list, y_validate_list)
            # print('average precision_mean', precision_mean,
            #       'average recall_mean', recall_mean,
            #       'average accuracy_mean', accuracy_mean,
            #       'average f1_mean', f1_mean,)
            # return f1_mean, accuracy_mean

        # return max_f1,best_params

def eval_score(model_name, clf, X_train, y_train, cv_num=10):
    print(model_name + ' Training the model :')
    accuracy = cross_val_score(clf, X_train, y_train, cv=cv_num, scoring='accuracy')
    precision = cross_val_score(clf, X_train, y_train, cv=cv_num, scoring='precision')
    recall = cross_val_score(clf, X_train, y_train, cv=cv_num, scoring='recall')
    f1 = cross_val_score(clf, X_train, y_train, cv=cv_num, scoring=make_scorer(f1_score, average='micro'))
    roc = cross_val_score(clf, X_train, y_train, cv=cv_num, scoring='roc_auc')
    print(model_name, " 10 cross validation result")
    print("[INFO] ACC:" + str(accuracy.mean()))
    print("[INFO] AUC:" + str(roc.mean()))
    print("[INFO] precision:" + str(precision.mean()))
    print("[INFO] recall:" + str(recall.mean()))
    print("[INFO] f1:" + str(f1.mean()))
def select_algorithm_param(algorithm_name, data_path):
    if algorithm_name == 'KNN':
        best_micro_score, best_param = selectKNNParam(data_path)
    elif algorithm_name == 'SVM':
        best_micro_score, best_param = selectSVMParam(data_path)
    elif algorithm_name == 'LR':
        best_micro_score, best_param = selectLRParam(data_path)
    elif algorithm_name == 'RF':
        best_micro_score, best_param = selectRFParam(data_path)
    elif algorithm_name == 'NBC':
        best_micro_score, best_param = selectNBCParam(data_path)
    elif algorithm_name == 'DTC':
        best_micro_score, best_param = selectDTCParam(data_path)
    elif algorithm_name == 'GBDT':
        best_micro_score, best_param = selectGBDTParam(data_path)
    elif algorithm_name == 'XGBOOST':
        best_micro_score, best_param = selectXGBoostParam(data_path)
    elif algorithm_name == 'MLP':
        best_micro_score, best_param = selectMLPParam(data_path)
    return best_micro_score, best_param
def test_all_oberservation_time():
    import datetime
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)

    # choose the observation time point
    times = ['30d','7d','1d','3h']
    # times = ['7d', '1d', '3h']
    # times = ['1d', '3h']
    # times = ['3h']

    # choose the Algorithms
    # method = 'KNN'
    # # method = 'SVM'
    # method = 'LR'
    method = 'RF'
    # method = 'NBC'
    # method = 'DTC'
    # method = 'GBDT'
    # method = 'XGBOOST'
    # method = 'MLP'
    print(method)
    for t in times:
        read_file_path = './dataset/' + 'issue_data_' + t + '_normal' + '.csv'
        # read_file_path = './dataset/Kikas/' + 'all_21f_issue_data_' + t + '_named' + '.csv'
        micro_score, param = select_algorithm_param(method,read_file_path)
        print(method, t, param)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)


# training-test sliding validation
# evaluation metrics include macro-F1 score, micro-F1 score, macro-precision score, and macro-recall
def get_model_train_by_slide(model,x_train_list, y_train_list, x_validate_list, y_validate_list):
    accuracy = []
    micro_precision = []
    micro_recall = []
    macro_precision = []
    macro_recall = []
    weight_precision = []
    weight_recall = []
    weight_f1 = []
    macro_f1 = []
    micro_f1 = []
    f1 = []
    cohen_kappa = []
    auc = []
    total_num = len(x_train_list)
    importance_rank_list = []
    for i in range(0, total_num):
        x_train = x_train_list[i]
        y_train = y_train_list[i]
        x_validate = x_validate_list[i]
        y_validate = y_validate_list[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_validate)
        # Calculating metrics
        accuracy.append(accuracy_score(y_validate, y_pred))
        micro_precision.append(precision_score(y_validate, y_pred, average='micro'))
        micro_recall.append(recall_score(y_validate, y_pred, average='micro'))
        macro_precision.append(precision_score(y_validate, y_pred, average='macro'))
        macro_recall.append(recall_score(y_validate, y_pred, average='macro'))
        weight_precision.append(precision_score(y_validate, y_pred, average='weighted'))
        weight_recall.append(recall_score(y_validate, y_pred, average='weighted'))
        weight_f1.append(f1_score(y_validate, y_pred, average='weighted'))
        macro_f1.append(f1_score(y_validate, y_pred, average='macro'))
        micro_f1.append(f1_score(y_validate, y_pred, average='micro'))
        ret = classification_report(y_validate, y_pred, digits=4)
        print(ret)
        cohen_kappa.append(cohen_kappa_score(y_validate, y_pred))

    print(weight_f1)
    accuracy_mean = mean(accuracy)
    micro_precision_mean = mean(micro_precision)
    micro_recall_mean = mean(micro_recall)

    macro_precision_mean = mean(macro_precision)
    macro_recall_mean = mean(macro_recall)
    weight_recall_mean = mean(weight_recall)
    weight_precision_mean = mean(weight_precision)
    macro_f1_mean = mean(macro_f1)
    micro_f1_mean = mean(micro_f1)
    weight_f1_mean = mean(weight_f1)
    cohen_kappa_mean = mean(cohen_kappa)

    return accuracy_mean, micro_precision_mean, micro_recall_mean,macro_precision_mean, macro_recall_mean,weight_precision_mean,weight_recall_mean, weight_f1_mean, macro_f1_mean,micro_f1_mean, cohen_kappa_mean


# training-test sliding validation
# evaluation metrics include F1, Accuracy
def get_model_train_by_slide_with_binary(model,x_train_list, y_train_list, x_validate_list, y_validate_list):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    cohen_kappa = []
    total_num = len(x_train_list)
    for i in range(0, total_num):
        x_train = x_train_list[i]
        y_train = y_train_list[i]
        x_validate = x_validate_list[i]
        y_validate = y_validate_list[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_validate)
        # Calculating metrics
        accuracy.append(accuracy_score(y_validate, y_pred))
        precision.append(precision_score(y_validate, y_pred, average='binary'))
        recall.append(recall_score(y_validate, y_pred, average='binary'))
        f1.append(f1_score(y_validate, y_pred, average='binary'))
        ret = classification_report(y_validate, y_pred, digits=4)
        print(ret)
        cohen_kappa.append(cohen_kappa_score(y_validate, y_pred))

    accuracy_mean = mean(accuracy)
    precision_mean = mean(precision)
    recall_mean = mean(recall)
    f1_mean = mean(f1)
    print('AUC', accuracy_mean)
    print('F1', f1_mean)
    return precision_mean,recall_mean,accuracy_mean,f1_mean

def savefile(filepath,filename,node):
    filepath = filepath + filename
    with open(filepath, 'w') as file_object:
        json.dump(node, file_object)

if __name__ == '__main__':
    test_all_oberservation_time()