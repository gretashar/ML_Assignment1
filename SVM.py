import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from balance import balance_test_over
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
from sklearn.model_selection import ShuffleSplit
import time
import sklearn

# %matplotlib inline

import warnings

warnings.filterwarnings('ignore')

def svm_test(df, output, title):    
    
    if title == "Stroke Prediction":
        df = df.replace('N/A', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.drop(['id'], axis=1)
        # X, y, X_train, y_train, X_test, y_test = balance_test_over(df)
        # X_train, X_test, y_train_normal, y_test_normal = train_test_split(X, y, test_size = 0.2, random_state = 42)
        # X, y, X_train, y_train, X_test, y_test = balance_test_over(df)
        X, y, X_train_over, y_train_over, X_test_over, y_test_over = balance_test_over(df)
        # print(X_test_over)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    else:
        X = df.drop([output], axis=1)

        y = df[output]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    
    
    # X = df.drop([output], axis=1)
    # y = df[output]
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    cols = X_train.columns
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    kfold=KFold(n_splits=10, shuffle=True, random_state=0)

    # DEFAULT PARAMETERS: C=1.0, KERNEL=RBF, GAMMA=AUTO
    # instantiate classifier with default hyperparameters
    svc=SVC() 

    # fit classifier to training set
    svc.fit(X_train,y_train)

    # make predictions on test set
    y_pred=svc.predict(X_test)
    y_train_pred=svc.predict(X_train)

    cross_vale_rbf=cross_val_score(svc, X, y=y, cv=kfold)
    # compute and print accuracy score
    print(title+' Model accuracy score with rbf kernel and C=1.0: {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
    print(title+' Training accuracy score with rbf kernel and C=1.0: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred)))
    print(title+' Training CV accuracy score with rbf kernel and C=1.0: {0:0.4f}'. format(cross_vale_rbf.mean()))
    print(title+' Training set score: {:.4f}'.format(svc.score(X_train, y_train)))

    # print(cross_vale_rbf))

    # #PARAMETERS: C=10.0, KERNEL=RBF, GAMMA=AUTO
    # # instantiate classifier with rbf kernel and C=100
    # svc=SVC(C=10) 
    # # fit classifier to training set
    # svc.fit(X_train,y_train)

    # # make predictions on test set
    # y_pred=svc.predict(X_test)


    # # compute and print accuracy score
    # print(title+' Model accuracy score with rbf kernel and C=10.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))




    # # PARAMETERS: C=100.0, KERNEL=RBF, GAMMA=AUTO
    # # instantiate classifier with rbf kernel and C=100
    # svc=SVC(C=100) 

    # # fit classifier to training set
    # svc.fit(X_train,y_train)


    # # make predictions on test set
    # y_pred=svc.predict(X_test)


    # # compute and print accuracy score
    # print(title+' Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))







    # instantiate classifier with linear kernel and C=1.0
    linear_svc=SVC(kernel='linear', C=1.0) 


    # fit classifier to training set
    linear_svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred_test=linear_svc.predict(X_test)


    # compute and print accuracy score
    print(title+' Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
    cross_vale_linear=cross_val_score(linear_svc, X, y=y, cv=kfold)
    print(title+' Training CV accuracy score with linear kernel and C=1.0: {0:0.4f}'. format(cross_vale_linear.mean()))
    print(title+' Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))


    
    # # instantiate classifier with linear kernel and C=100.0
    # linear_svc100=SVC(kernel='linear', C=100.0) 


    # # fit classifier to training set
    # linear_svc100.fit(X_train, y_train)


    # # make predictions on test set
    # y_pred=linear_svc100.predict(X_test)


    # # compute and print accuracy score
    # print(title+' Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



    # y_pred_train = linear_svc.predict(X_train)

    # print(title+' Training-set accuracy score: {0:0.4f}'. format(balanced_accuracy_score(y_train, y_pred_train)))

    # print(title+' Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))

    # print(title+' Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))


    # # check class distribution in test set
    # print(y_test.value_counts())
    # null_accuracy = (3306/(3306+274))

    # print(title+' Null accuracy score: {0:0.4f}'. format(null_accuracy))
    
    
    
    
    
    # instantiate classifier with polynomial kernel and C=1.0
    poly_svc=SVC(kernel='poly', C=1.0) 


    # fit classifier to training set
    poly_svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred=poly_svc.predict(X_test)


    # compute and print accuracy score
    print(title+' Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
    cross_vale_poly=cross_val_score(poly_svc, X, y=y, cv=kfold)
    print(title+' Training CV accuracy score with poly kernel and C=1.0: {0:0.4f}'. format(cross_vale_poly.mean()))
    print(title+' Training set score: {:.4f}'.format(poly_svc.score(X_train, y_train)))
    
    
    # # instantiate classifier with polynomial kernel and C=100.0
    # poly_svc100=SVC(kernel='poly', C=100.0) 


    # # fit classifier to training set
    # poly_svc100.fit(X_train, y_train)


    # # make predictions on test set
    # y_pred=poly_svc100.predict(X_test)


    # # compute and print accuracy score
    # print(title+' Model accuracy score with polynomial kernel and C=100.0 : {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
        
        
    
    
    # instantiate classifier with sigmoid kernel and C=1.0
    sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


    # fit classifier to training set
    sigmoid_svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred=sigmoid_svc.predict(X_test)


    # compute and print accuracy score
    print(title+' Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
    cross_vale_sig=cross_val_score(sigmoid_svc, X, y=y, cv=kfold)
    print(title+' Training CV accuracy score with poly kernel and C=1.0: {0:0.4f}'. format(cross_vale_sig.mean()))
    print(title+' Training set score: {:.4f}'.format(sigmoid_svc.score(X_train, y_train)))  
        
    
    #https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    labels = ['rbf', 'poly', 'linear', 'sigmoid']
    men_means = [svc.score(X_train, y_train), poly_svc.score(X_train, y_train), linear_svc.score(X_train, y_train), sigmoid_svc.score(X_train, y_train)]
    women_means = [cross_vale_rbf.mean(), cross_vale_poly.mean(), cross_vale_linear.mean(), cross_vale_sig.mean()]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='Training Score')
    rects2 = ax.bar(x + width/2, women_means, width, label='Cross-Validation Score')
    
    ax.set_ylim([0, 1.2])
    ax.set_yticks([ x*0.2 for x in range(6)])
    ax.set_ylabel('Scores')
    ax.set_title('Heart Attack Prediction SVM Kernels')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=5)
    ax.bar_label(rects2, padding=5)

    # fig.tight_layout()
    # plt.legend(loc="best")
    plt.savefig('./plots/'+title+'bar.png')  
        
    plt.clf()
    # # instantiate classifier with sigmoid kernel and C=100.0
    # sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


    # # fit classifier to training set
    # sigmoid_svc100.fit(X_train,y_train)


    # # make predictions on test set
    # y_pred=sigmoid_svc100.predict(X_test)


    # # compute and print accuracy score
    # print(title+' Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(balanced_accuracy_score(y_test, y_pred)))
      
      
      
      
      
    # # Print the Confusion Matrix and slice it into four pieces

    

    # cm = confusion_matrix(y_test, y_pred_test)

    # print('Confusion matrix\n\n', cm)

    # print('\nTrue Positives(TP) = ', cm[0,0])

    # print('\nTrue Negatives(TN) = ', cm[1,1])

    # print('\nFalse Positives(FP) = ', cm[0,1])

    # print('\nFalse Negatives(FN) = ', cm[1,0])
    
    
    
    # # visualize confusion matrix with seaborn heatmap
    # cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
    #                                 index=['Predict Positive:1', 'Predict Negative:0'])

    # sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # plt.savefig('./plots/'+title+'confusion_svm.png')  
        
    # plt.clf()
    
    
    # print(classification_report(y_test, y_pred_test))
    # TP = cm[0,0]
    # TN = cm[1,1]
    # FP = cm[0,1]
    # FN = cm[1,0]
        
    # # print classification accuracy
    # classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    # print(title+' Classification accuracy : {0:0.4f}'.format(classification_accuracy))
    
 
    # # print classification error
    # classification_error = (FP + FN) / float(TP + TN + FP + FN)
    # print(title+' Classification error : {0:0.4f}'.format(classification_error))
    
    
    # # print precision score
    # precision = TP / float(TP + FP)
    # print(title+' Precision : {0:0.4f}'.format(precision))
    
    
    # recall = TP / float(TP + FN)
    # print(title+' Recall or Sensitivity : {0:0.4f}'.format(recall))
    
    # true_positive_rate = TP / float(TP + FN)
    # print(title+' True Positive Rate : {0:0.4f}'.format(true_positive_rate))
    
    
    # false_positive_rate = FP / float(FP + TN)
    # print(title+' False Positive Rate : {0:0.4f}'.format(false_positive_rate))
    
    # specificity = TN / (TN + FP)
    # print(title+' Specificity : {0:0.4f}'.format(specificity))
    
    
    
    
    
    
    # #ROC Curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

    # plt.figure(figsize=(6,4))

    # plt.plot(fpr, tpr, linewidth=2)

    # plt.plot([0,1], [0,1], 'k--' )

    # plt.rcParams['font.size'] = 12

    # plt.title(title+' ROC Curve')

    # plt.xlabel('False Positive Rate (1 - Specificity)')

    # plt.ylabel('True Positive Rate (Sensitivity)')

    # plt.savefig('./plots/'+title+'ROC_SVM.png') 
        
    # plt.clf()
    
    # # compute ROC AUC
    # ROC_AUC = roc_auc_score(y_test, y_pred_test)
    # print(title+' ROC AUC : {:.4f}'.format(ROC_AUC))
    
    
    
    
    
     
     

    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



    # calculate cross-validated ROC AUC 
    Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10, scoring='roc_auc').mean()
    print(title+' Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
    
    
    kfold=KFold(n_splits=5, shuffle=True, random_state=0)
    linear_svc=SVC(kernel='linear')
    linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)
    
    # print cross-validation scores with linear kernel
    print(title+' Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))
    
    # print average cross-validation score with linear kernel
    print(title+' Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))
    
    rbf_svc=SVC(kernel='rbf')
    rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
    
    # print cross-validation scores with rbf kernel
    print(title+' Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))

    # print average cross-validation score with rbf kernel
    print(title+' Average stratified cross-validation score with rbf kernel:{:.4f}'.format(rbf_scores.mean()))
            
             
    clf = SVC() 
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
        

        # instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
        svc=SVC() 



        # declare parameters for hyperparameter tuning
        parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
                    {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                    {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]} 
                    ]
        





        grid_search = GridSearchCV(estimator = svc,  
                                param_grid = parameters,
                                scoring = 'balanced_accuracy',
                                verbose=0, return_train_score=True)


        grid_search.fit(X_train, y_train)
            



        # examine the best model

        # # best score achieved during the GridSearchCV
        # print(title+' GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


        # # print parameters that give the best results
        # print(title+' Parameters that give the best results :','\n\n', (grid_search.best_params_))


        # # print estimator that was chosen by the GridSearch
        # print('\n\n'+title+' Estimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


        # # calculate GridSearch CV score on test set
        # print(title+' GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
        
        
        

        
        # Cross validation with 50 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        # cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        
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
        
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        title1 = "Accuracy Learning Curves (SVM) for " + title
        
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

        plt.savefig('./plots/'+title+' svm_learning_balanced_curve.png')  
        plt.clf()
        
        # title1 = "F1 Learning Curves (SVM) for " + title
        # fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        # plot_learning_curve(
        #     estimator,
        #     title1,
        #     X,
        #     y,
        #     axes=axes,
        #     ylim=(0.5, 1.01),
   
        #     n_jobs=4,
        #     scoring="f1",
        # )

        # plt.savefig('./plots/'+title+' svm_learning_f1_curve_.png')  
        # plt.clf()
        
    
        
        param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
            SVC(kernel="rbf"),
            X,
            y,
            param_name="gamma",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"Gamma Validation (rbf)", train_scores, test_scores, param_range, "SVM Gamma ")
        
        
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
            SVC(kernel="rbf"),
            X,
            y,
            param_name="C",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"C Validation (rbf)", train_scores, test_scores, param_range, "SVM C (rbf)")
        
        
        
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
            SVC(kernel="poly"),
            X,
            y,
            param_name="C",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"C Validation (poly)", train_scores, test_scores, param_range, "SVM C (poly)")
        
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
            SVC(kernel="linear"),
            X,
            y,
            param_name="C",
            param_range=param_range,
            scoring="balanced_accuracy",
            n_jobs=2,
        )
        
        single_valid(title+"C Validation (linear)", train_scores, test_scores, param_range, "SVM C (linear)")    


if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    svm_test(df, "output", "Heart Failure Prediction")
    # data = "./data/stroke.csv"
    # df = pd.read_csv(data)
    # svm_test(df, "stroke", "Stroke Prediction")