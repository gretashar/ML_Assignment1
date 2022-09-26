import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns # for statistical data visualization
# from DT import df_test


#Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def balance_test_over(df_train):

    
    X = df_train.drop(['stroke'], axis=1)
    y = df_train['stroke']
    X2=X
    y2=y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    

    encoder = ce.OrdinalEncoder(cols=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    X2 = encoder.transform(X2)

    
    for df1 in [X_train, X_test]:
        for col in X_train.columns:
            col_median=X_train[col].median()
            df1[col].fillna(col_median, inplace=True)      
            
    for col in X2.columns:
        col_median=X2[col].median()
        X2[col].fillna(col_median, inplace=True)      
    # # model = XGBClassifier()
    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # print("")
    # print("*************************************** --  Data BEFORE Sampling --  ***********************************************")
    # print("")
    # print("Accuracy BEFORE Sampling: %.2f%%" % (accuracy * 100.0))


    # conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # print("Confusion Matrix BEFORE: ", pd.DataFrame(confusion_matrix(y_test, y_pred)))
    
    # cm = confusion_matrix(y_test, y_pred)

    # print('Confusion matrix\n\n', cm)

    # print('\nTrue Positives(TP) BEFORE Sampling = ', cm[0,0])

    # print('\nTrue Negatives(TN) BEFORE Sampling = ', cm[1,1])

    # print('\nFalse Positives(FP) BEFORE Sampling = ', cm[0,1])

    # print('\nFalse Negatives(FN) BEFORE Sampling = ', cm[1,0])
    
    
    
    # # visualize confusion matrix with seaborn heatmap
    # cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
    #                                 index=['Predict Positive:1', 'Predict Negative:0'])

    # sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # plt.savefig('./plots/confusion_balance_before_over.png')  
    # plt.clf()
    
    # print("")
    # print("*************************************** --  Data AFTER Oversampling --  ***********************************************")
    # print("")
    
    # print("F1 Score AFTER Oversampling: ", f1_score(y_test, y_pred))
    # print("Recall Score AFTER Oversampling: ", recall_score(y_test, y_pred))
    
    
    
    
    
    
    
    #Oversample Minority Class
    X = pd.concat([X_train, y_train], axis=1)
    # print(X.head())
    
    # separate minority and majority classes
    not_stroke = X[X.stroke==0]
    stroke = X[X.stroke==1]
    
    
    # upsample minority
    fraud_upsampled = resample(stroke,
                            replace=True, # sample with replacement
                            n_samples=len(not_stroke), # match number in majority class
                            random_state=27) # reproducible results


    # combine majority and upsampled minority
    upsampled = pd.concat([not_stroke, fraud_upsampled])

    # check new class counts
    print("AFTER Oversampling Value Counts: ",upsampled.stroke.value_counts())

    y_train = upsampled.stroke
    X_train = upsampled.drop('stroke', axis=1)
    
    # X = pd.concat([X_train, X_test])
    # y = pd.concat([y_train, y_test])
    
    return X2, y2, X_train, y_train, X_test, y_test
    
    
    # model = XGBClassifier()
    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy AFTER Oversampling: %.2f%%" % (accuracy * 100.0))
    
    
    
    # print("F1 Score AFTER Oversampling: ", f1_score(y_test, y_pred))
    # print("Recall Score AFTER Oversampling: ", recall_score(y_test, y_pred))
    
    # # print("Confusion Matrix AFTER: ", pd.DataFrame(confusion_matrix(y_test, y_pred)))
    # cm = confusion_matrix(y_test, y_pred)
    # cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
    #                                 index=['Predict Positive:1', 'Predict Negative:0'])

    # sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # plt.savefig('./plots/confusion_balance_after_over.png')  
    # plt.clf()
    # print('Confusion matrix AFTER Oversampling\n\n', cm)

    # print('\nTrue Positives(TP) AFTER Oversample = ', cm[0,0])

    # print('\nTrue Negatives(TN) AFTER Oversample = ', cm[1,1])

    # print('\nFalse Positives(FP) AFTER Oversample = ', cm[0,1])

    # print('\nFalse Negatives(FN) AFTER Oversample = ', cm[1,0])
    
    
    
    
    
    
def balance_test_under(df_train):

    
    X = df_train.drop(['stroke'], axis=1)
    y = df_train['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    

    encoder = ce.OrdinalEncoder(cols=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

    X_train = encoder.fit_transform(X_train)

    X_test = encoder.transform(X_test)
    # print(X[-1])
    
    for df1 in [X_train, X_test]:
        for col in X_train.columns:
            col_median=X_train[col].median()
            df1[col].fillna(col_median, inplace=True)      
    
    
    #Oversample Minority Class
    X = pd.concat([X_train, y_train], axis=1)
    # print(X.head())
    
    # separate minority and majority classes
    not_stroke = X[X.stroke==0]
    stroke = X[X.stroke==1]
    

    # downsample majority
    not_stroke_downsampled = resample(not_stroke,
                                    replace = False, # sample without replacement
                                    n_samples = len(stroke), # match minority n
                                    random_state = 27) # reproducible results

    # combine minority and downsampled majority
    downsampled = pd.concat([not_stroke_downsampled, stroke])

    # checking counts
    print(downsampled.stroke.value_counts())
        
    
    
    # trying logistic regression again with the undersampled dataset
    y_train = downsampled.stroke
    X_train = downsampled.drop('stroke', axis=1)
    
    return X_train, y_train, X_test, y_test
       
    # model = XGBClassifier()
    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    
    # print("")
    # print("*************************************** --  Data AFTER Undersampling --  ***********************************************")
    # print("")
    # print("Accuracy AFTER Undersampling: %.2f%%" % (accuracy * 100.0))
    
    
    
    # print("F1 Score AFTER Undersampling: ", f1_score(y_test, y_pred))
    # print("Recall Score AFTER Undersampling: ", recall_score(y_test, y_pred))
    
    # print("Confusion Matrix AFTER Undersampling: ", pd.DataFrame(confusion_matrix(y_test, y_pred)))
    
    # cm = confusion_matrix(y_test, y_pred)
    # cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
    #                                 index=['Predict Positive:1', 'Predict Negative:0'])

    # sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # plt.savefig('./plots/confusion_balance_after_under.png')  
    # plt.clf()
    # print('Confusion matrix AFTER Undersampling\n\n', cm)

    # print('\nTrue Positives(TP) AFTER Undersampling = ', cm[0,0])

    # print('\nTrue Negatives(TN) AFTER Undersampling = ', cm[1,1])

    # print('\nFalse Positives(FP) AFTER Undersampling = ', cm[0,1])

    # print('\nFalse Negatives(FN) AFTER Undersampling = ', cm[1,0])
    
    
if __name__ == "__main__":
    
    data = "./data/stroke.csv"
    df = pd.read_csv(data)
    df = df.replace('N/A', np.nan)
    df = df.replace('Unknown', np.nan)
    df = df.drop(['id'], axis=1)
    # print(df.tail())
            
    target_count = df["stroke"].value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')


    plot = target_count.plot(kind='bar', title='Count (target)')
    fig = plot.get_figure()
    fig.savefig('./plots/stroke_output.png')
    plt.clf()
    balance_test_over(df)
    balance_test_under(df)