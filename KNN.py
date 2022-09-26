import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from balance import balance_test_over
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
from sklearn.model_selection import ShuffleSplit
import time


warnings.filterwarnings('ignore')

#Source: https://www.kaggle.com/code/melihkanbay/knn-best-parameters-gridsearchcv
#https://www.kaggle.com/code/prashant111/knn-classifier-tutorial
def knn_test(df, output, title):
   
    
    if title == "Stroke Prediction":
        df = df.replace('N/A', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.drop(['id'], axis=1)
        X, y, X_train_over, y_train_over, X_test_over, y_test_over = balance_test_over(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    else:
        X = df.drop([output], axis=1)

        y = df[output]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


    cols = X_train.columns
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    
    clf = KNeighborsClassifier()
    if title == 'Stroke Prediction':
        X_train_over = scaler.fit_transform(X_train_over)
        X_test_over = scaler.transform(X_test_over)
        X_train_over = pd.DataFrame(X_train_over, columns=[cols])
        X_test_over = pd.DataFrame(X_test_over, columns=[cols])
        clf.fit(X_train_over,y_train_over)
        y_pred = clf.predict(X_test_over)
        y_train_pred = clf.predict(X_train_over)
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test_over, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train_over, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train_over, y_train_over)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test_over, y_test_over)))
    else:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)        
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train, y_train)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test, y_test)))    
    
    

     
    if title != "Stroke Predictionplaceholder":    
      

        # instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
        knn=KNeighborsClassifier(metric='minkowski') 



        # declare parameters for hyperparameter tuning
        params = {'weights': ['uniform', 'distance'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'leaf_size': [5, 10, 15, 20, 25, 30, 35, 40 , 45, 50]}



        grid_search = GridSearchCV(estimator = knn,  
                                param_grid = params,
                                scoring = 'balanced_accuracy',
                                cv = 5,
                                verbose=0, return_train_score=True)


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
        title1 = "Balanced Accuracy Learning Curves (KNN) for " + title
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

        plt.savefig('./plots/'+title+' knn_learning_balanced_accuracy_curve_.png')  
        plt.clf()
        
        
        
        param_range = range(1,101)
        train_scores, test_scores = validation_curve(
            KNeighborsClassifier(metric='minkowski', weights = 'uniform'),
            X,
            y,
            param_name="n_neighbors",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"K Value Validation (uniform)", train_scores, test_scores, param_range, "KNN n-neighbors (uniform)")
        
        
        param_range = range(1,101)
        train_scores, test_scores = validation_curve(
            KNeighborsClassifier(metric='minkowski', weights = 'distance'),
            X,
            y,
            param_name="n_neighbors",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"K Value Validation (distance)", train_scores, test_scores, param_range, "KNN n-neighbors (distance)")
    

if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    knn_test(df, "output", "Heart Failure Prediction")
    data = "./data/stroke.csv"
    df = pd.read_csv(data)
    knn_test(df, "stroke", "Stroke Prediction")