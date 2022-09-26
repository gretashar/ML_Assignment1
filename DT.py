from time import time
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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import category_encoders as ce
from balance import balance_test_over, balance_test_under
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve, validation_curve_plot
import time
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
# import graphviz 

#Source: https://www.kaggle.com/code/mabalogun/titanic-gridsearchcv-with-decisiontreeclassifier
# Source: https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial/notebook
def df_test(df, output, title):
    
    X_test_over = None
    if title == "Stroke Prediction":
        df = df.replace('N/A', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.drop(['id'], axis=1)
        X, y, X_train_over, y_train_over, X_test_over, y_test_over = balance_test_over(df)
        # print(X_test_over)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    else:
        X = df.drop([output], axis=1)

        y = df[output]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # print(X_train.dtypes)

    # params = {'max_depth': [2,4,6,8,10,12,13,14,15],
    #      'min_samples_split': [2,3,4,5,6,7,8,9,10],
    #      'min_samples_leaf': [1,2,3,4,5,6,7,8,9.10]}

    # clf = tree.DecisionTreeClassifier(criterion='gini')
    # gcv = GridSearchCV(estimator=clf,param_grid=params, return_train_score=True, scoring = "balanced_accuracy")
    # gcv.fit(X_train,y_train)
    # print("HERE:")
    # print(f'Best parameters {gcv.best_params_}')

    # model = gcv.best_estimator_
    # start_gini_train = time.time()
    # model.fit(X_train,y_train)
    # end_gini_train = time.time()
    # gini_train_time = end_gini_train-start_gini_train
    
    # start_gini_train_predict = time.time()
    # y_train_pred = model.predict(X_train)
    # end_gini_train_predict = time.time()
    # gini_train_predict_time = end_gini_train_predict-start_gini_train_predict    
    
    # start_gini_test_predict = time.time()
    # y_test_pred = model.predict(X_test)
    # end_gini_test_predict = time.time()
    # gini_test_predict_time = end_gini_test_predict-start_gini_test_predict
    
    # print("\n*******************************************************\n")
    
    # print(title+' Timing for Gini Training: {:.6f}'. format(gini_train_time))
    # print(title+' Timing for Gini Train Predict: {:.6f}'. format(gini_train_predict_time))
    # print(title+' Timing for Gini Test Predict: {:.6f}'. format(gini_test_predict_time))
    
    # print("\n*******************************************************\n")
    
    
    
    # df_grid_gini = pd.DataFrame(gcv.cv_results_)

    # results = ['mean_test_score',
    #        'mean_train_score',
    #        'std_test_score', 
    #        'std_train_score']
    
    # validation_curve_plot(df_grid_gini, params, results, title+ " Tree (Gini)")


    # print(title+' Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))

    # y_pred_train_gini = y_train_pred

    
    # print(title+' Training-set accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
    
    # print(title+' Training set score with criterion gini index: {:.4f}'.format(model.score(X_train, y_train)))

    # print(title+' Test set score with criterion gini index: {:.4f}'.format(model.score(X_test, y_test)))


    # plt.figure(figsize=(12,8))
    # plt.title(title + " Descision Tree with criterion gini index")
    # tree.plot_tree(model.fit(X_train, y_train)) 
    # plt.savefig('./plots/'+title+' tree with criterion gini index.png')  
    
    # plt.clf()
    
    print("TESTTTTT")    
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
    if title == 'Stroke Prediction':
        clf.fit(X_train_over,y_train_over)
        y_pred = clf.predict(X_test_over)
        y_train_pred = clf.predict(X_train_over)
        cm = confusion_matrix(y_test_over, y_pred)
        print('Confusion matrix for '+title+'with criterion entropy:\n\n', cm)
        print(classification_report(y_test_over, y_pred))
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test_over, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train_over, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train_over, y_train_over)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test_over, y_test_over)))
    else:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
        cm = confusion_matrix(y_test, y_pred)
        
        print('Confusion matrix for '+title+'with criterion entropy:\n\n', cm)
        print(classification_report(y_test, y_pred))           
        print(title+' Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
        print(title+' Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
        print(title+' Training set score with criterion entropy: {:.4f}'.format(clf.score(X_train, y_train)))
        print(title+' Test set score with criterion entropy: {:.4f}'.format(clf.score(X_test, y_test)))

    print("TESTTTTT")  


    
    
    
    params = {'max_depth': range(2,40,2),
         'min_samples_split': [2,3,4,5,6,7,8,9,10],
         'min_samples_leaf': range(2,40,2)}

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    
    gcv = GridSearchCV(estimator=clf,param_grid=params, return_train_score=True, scoring = "balanced_accuracy")
    gcv.fit(X_train,y_train)




    # # fit the model
    model = gcv.best_estimator_
    # model.fit(X_train,y_train)
    # y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)
    
    start_entropy_train = time.time()
    model.fit(X_train,y_train)
    end_entropy_train = time.time()
    entropy_train_time = end_entropy_train-start_entropy_train
    
    start_entropy_train_predict = time.time()
    y_train_pred = model.predict(X_train)
    end_entropy_train_predict = time.time()
    entropy_train_predict_time = end_entropy_train_predict-start_entropy_train_predict    
    
    start_entropy_test_predict = time.time()
    y_test_pred = model.predict(X_test)
    end_entropy_test_predict = time.time()
    entropy_test_predict_time = end_entropy_test_predict-start_entropy_test_predict
    
    print("\n*******************************************************\n")
    
    print(title+' Timing for Entropy Training: {:.6f}'. format(entropy_train_time))
    print(title+' Timing for Entropy Train Predict: {:.6f}'. format(entropy_train_predict_time))
    print(title+' Timing for Entropy Test Predict: {:.6f}'. format(entropy_test_predict_time))
    
    print("\n*******************************************************\n")
    
    
    
    y_pred_en = y_test_pred
    
    print("Best score: %0.4f" % gcv.best_score_)
    print(title+"Using the following parameters:")
    print(gcv.best_params_)

    
    print(title+' GridSearch Model balanced accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred_en)))
    print(title+' GridSearch Training-set accuracy score with criterion entropy: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_train_pred))) 
    print(title+' GridSearch Training set score with criterion entropy: {:.4f}'.format(model.score(X_train, y_train)))
    print(title+' GridSearch Test set score with criterion entropy: {:.4f}'.format(model.score(X_test, y_test)))
    
    plt.figure(figsize=(12,8))
    tree.plot_tree(model.fit(X_train, y_train)) 
    plt.savefig('./plots/'+title+' tree_entropy with criterion entropyGridSearch.png')  
    plt.clf()



    # Print the Confusion Matrix and slice it into four pieces
    cm = confusion_matrix(y_test, y_pred_en)

    print('Confusion matrix for '+title+'with criterion entropy:\n\n', cm)
    print(classification_report(y_test, y_pred_en))
    
    
    plt.figure(figsize=(6,4))
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                    index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.savefig('./plots/'+title+' confusion_dt with criterion entropy.png')  
    plt.clf()


    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    param_range = range(2,40,2)
    train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(criterion='entropy'),
            X,
            y,
            param_name="max_depth",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
    single_valid(title+"DT Max Depth Validation", train_scores, test_scores, param_range, "Decision Tree max_depth ")
    
    
    param_range = range(2,40,2)
    train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(criterion='entropy'),
            X,
            y,
            param_name="min_samples_leaf",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
    single_valid(title+"DT Min Sample Leaf Validation", train_scores, test_scores, param_range, "Decision Tree min_samples_leaf ")


    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    title1 = "Accuracy Learning Curves (Decision Tree) for " + title
    estimator = model
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

    plt.savefig('./plots/'+title+' balanced_dt_learning_curve.png')  
    plt.clf()
    # title1 = "F1 Learning Curves (Decision Tree) for " + title
    # fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    # plot_learning_curve(
    #     estimator,
    #     title1,
    #     X,
    #     y,
    #     axes=axes,
    #     ylim=(0.5, 1.01),
    #     cv=cv,
    #     n_jobs=4,
    #     scoring="f1",
    # )

    # plt.savefig('./plots/'+title+' f1_dt_learning_curve.png')  
    # plt.clf()

if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    df_test(df, "output", "Heart Attack Prediction")
    data = "./data/stroke.csv"
    df = pd.read_csv(data)
    df_test(df, "stroke", "Stroke Prediction")