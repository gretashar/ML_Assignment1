import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import category_encoders as ce
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import cv
from balance import balance_test_over
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import time
from balance import balance_test_over
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
# import graphviz 

#Source: https://www.kaggle.com/code/bachnguyentfk/adaboost-hyperparameters-grid-search
#https://www.kaggle.com/code/prashant111/adaboost-classifier-tutorial/notebook

def boost_test(df, output, title):
    
    
    # X = df.drop(['output'], axis=1)

    # y = df['output']
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    # # print(X_train.dtypes)


    
    if title == "Stroke Prediction":
        df = df.replace('N/A', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.drop(['id'], axis=1)
        X, y, X_train_over, y_train_over, X_test_over, y_test_over = balance_test_over(df)
        # print(X_test_over)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        # X, y, X_train, y_train, X_test, y_test = balance_test_over(df)

    else:
        X = df.drop([output], axis=1)

        y = df[output]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


    cols = X_train.columns
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])


    # # Create adaboost classifer object
    # abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

    # # Train Adaboost Classifer
    # model1 = abc.fit(X_train, y_train)


    # #Predict the response for test dataset
    # y_pred = model1.predict(X_test)


    # # calculate and print model accuracy
    # print(title+"AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))


    # svc=SVC(probability=True, kernel='linear')


    # # create adaboost classifer object
    # abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1, random_state=0)


    # # train adaboost classifer
    # model2 = abc.fit(X_train, y_train)


    # # predict the response for test dataset
    # y_pred = model2.predict(X_test)


    # # calculate and print model accuracy
    # print("Model Accuracy with SVC Base Estimator:",accuracy_score(y_test, y_pred))
    

    clf = AdaBoostClassifier()
    if title == 'Stroke Prediction':
        X_train_over = scaler.fit_transform(X_train_over)
        X_test_over = scaler.transform(X_test_over)
        X_train_over = pd.DataFrame(X_train_over, columns=[cols])
        X_test_over = pd.DataFrame(X_test_over, columns=[cols])
        clf.fit(X_train_over,y_train_over)
        y_pred = clf.predict(X_test_over)
        y_train_pred = clf.predict(X_train_over)
        # cm = confusion_matrix(y_test_over, y_pred)
        # print('Confusion matrix for '+title+'with criterion entropy:\n\n', cm)
        # print(classification_report(y_test_over, y_pred))
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test_over, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train_over, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train_over, y_train_over)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test_over, y_test_over)))
    else:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
        # cm = confusion_matrix(y_test, y_pred)
        
        # print('Confusion matrix for '+title+'with criterion entropy:\n\n', cm)
        # print(classification_report(y_test, y_pred))           
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train, y_train)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test, y_test)))


     
    if title != "Stroke Prediction_placeholder":    
        
        
        # Cross_validated_ROC_AUC = cross_val_score(model1, X_train, y_train, cv=5, scoring='roc_auc').mean()

        # print(title+ ' Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
        
        
        
        # scores = cross_val_score(model1, X_train, y_train, cv = 10, scoring='accuracy')

        # print(title+' Cross-validation scores:{}'.format(scores))
        
        
        # # compute Average cross-validation score
        # print(title+' Average cross-validation score: {:.4f}'.format(scores.mean()))
        
        
            
            
        # define the model with default hyperparameters
        model = AdaBoostClassifier()

        # define the grid of values to search
        grid = dict()
        grid['n_estimators'] = [10, 50, 100, 200, 300, 400, 500]
        grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

        # define the evaluation procedure
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        # define the grid search procedure
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='balanced_accuracy')    
            

        # # instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
        # boost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()) 



        # # declare parameters for hyperparameter tuning
        # params = {'base_estimator__max_depth':[i for i in range(2,11,2)],
        #       'base_estimator__min_samples_leaf':[5,10],
        #       'n_estimators':[10,50,250,1000],
        #       'learning_rate':[0.01,0.1]}
        # grid_search = GridSearchCV(estimator = boost,  
        #                         param_grid = params,
        #                         scoring = 'balanced_accuracy',
        #                         cv = 5,
        #                         verbose=0, return_train_score=True)


        grid_search.fit(X_train, y_train)
        
        estimator = grid_search.best_estimator_   
        
        start_train = time.time()
        estimator.fit(X_train,y_train)
        end_train = time.time()
        train_time = end_train-start_train
        
        start_train_predict = time.time()
        y_train_pred = estimator.predict(X_train)
        end_train_predict = time.time()
        train_predict_time = end_train_predict-start_train_predict  
        
        start_test_predict = time.time()
        y_test_pred = estimator.predict(X_test)
        end_test_predict = time.time()
        test_predict_time = end_test_predict-start_test_predict
        
        print("\n*******************************************************\n")
        
        print(title+' Timing for Training: {:.6f}'. format(train_time))
        print(title+' Timing for Train Predict: {:.6f}'. format(train_predict_time))
        print(title+' Timing for Test Predict: {:.6f}'. format(test_predict_time))

        print("\n*******************************************************\n")
        print(title+' GridSearch Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_test_pred)))
        print(title+' GridSearch Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
        print(title+' GridSearch Training set score with criterion entropy: {:.4f}'.format(estimator.score(X_train, y_train)))
        print(title+' GridSearch Test set score with criterion entropy: {:.4f}'.format(estimator.score(X_test, y_test)))
        
        print("Best score: %0.4f" % grid_search.best_score_)
        print(title+"Using the following parameters:")
        print(grid_search.best_params_)


        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        title1 = "Accuracy Learning Curves (Adaboost) for " + title
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        plot_learning_curve(
            estimator,
            title1,
            X,
            y,
            axes=axes,
            ylim=(0.5, 1.01),

            n_jobs=4,
            scoring="accuracy",
        )

        plt.savefig('./plots/'+title+' boost_learning_accuracy_curve_.png')  
        plt.clf()
        
        param_range = [10, 50, 100, 200, 300, 400, 500]
        train_scores, test_scores = validation_curve(
            AdaBoostClassifier(),
            X,
            y,
            param_name="n_estimators",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        
        single_valid(title+"Adaboost n-estimators Validation (uniform)", train_scores, test_scores, param_range, "Adaboost n-estimators ")
        
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0]
        train_scores, test_scores = validation_curve(
            AdaBoostClassifier(),
            X,
            y,
            param_name="learning_rate",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"Adaboost learning_rate Validation (uniform)", train_scores, test_scores, param_range, "Adaboost learning_rate ")
        







if __name__ == "__main__":
    
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    boost_test(df, "output", "Heart Failure Prediction")
    data = "./data/stroke.csv"
    df = pd.read_csv(data)
    boost_test(df, "stroke", "Stroke Prediction")