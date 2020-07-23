import utils
import etl
from models_partc import *
from cross import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import seaborn
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features(filepath_train, filepath_test):
	#TODO: complete this
    events, mortality, feature_map = etl.read_csv(filepath_train)
    events_test = pd.read_csv(filepath_test+'events.csv')
    feature_map_test = pd.read_csv(filepath_test+'event_feature_map.csv')
    #deliverables_path = 'C:/Users/yyan/Downloads/'
    
    indx_date = etl.calculate_index_date(events, mortality, '')
    filtered_events = etl.filter_events(events, indx_date, '')
    aggregated_events = etl.aggregate_events(filtered_events, mortality, feature_map, deliverables_path)
    
    feature_count = aggregated_events.groupby(by=['feature_id']).count()
    n = 600
    selected_features = list(feature_count[feature_count['patient_id']>=n].index)
    aggregated_events = aggregated_events[aggregated_events['feature_id'].isin(selected_features)]
    
    df = aggregated_events.join(mortality.set_index('patient_id'),on='patient_id',lsuffix='',rsuffix='_r')
    patient_features=df.set_index('patient_id')[['feature_id', 'feature_value']].T.apply(tuple).to_frame()
    patient_features.columns=['features']
    patient_features=patient_features.groupby(by=['patient_id'])['features'].apply(np.array)
    mortality = df.fillna(0).drop_duplicates().set_index('patient_id')['label'].to_dict()
    s = aggregated_events.pivot_table(index='patient_id',columns='feature_id',values='feature_value').fillna(0)
    l = df[['patient_id','label']].fillna(0).drop_duplicates()
    
    df_test = events_test.join(feature_map_test.set_index('event_id'),on='event_id',lsuffix='',rsuffix='_r')
    sub_sum = df_test[df_test['event_id'].str.startswith(('DIAG','DRUG'))==True].groupby(by=['patient_id','idx']).sum()
    sub_count = df_test[df_test['event_id'].str.startswith(('LAB'))==True].groupby(by=['patient_id','idx']).count()
    sub_count = sub_count[['value']]
    columns=['patient_id', 'feature_id', 'feature_value']
    agg_events = pd.concat([sub_sum,sub_count]).reset_index()
    agg_events.columns = columns
    agg_events['feature_value'] = agg_events['feature_value']/agg_events.groupby(['feature_id'])['feature_value'].transform('max')
    
    #agg_events = agg_events[agg_events['feature_id'].isin(selected_features)]
    
    
    X_train = s
    Y_train = l.set_index('patient_id')      
    clf = LogisticRegression(penalty='l1')
    clf.fit(X_train,Y_train)
    coef = clf.coef_
    selected_features = pd.DataFrame(coef, columns = X_train.columns).columns.delete(0)
    
    X_train = X_train[selected_features]
    
    agg_events = agg_events[agg_events['feature_id'].isin(selected_features)].fillna(0)
    patient_features_test=agg_events.set_index('patient_id')[['feature_id', 'feature_value']].T.apply(tuple).to_frame()
    patient_features_test.columns=['features']
    patient_features_test=patient_features_test.groupby(by=['patient_id'])['features'].apply(np.array)
    X_test = agg_events.pivot_table(index='patient_id',columns='feature_id',values='feature_value').fillna(0)
    #X_test = X_test[selected_features]   
    
    deliverable = open('C:/Users/yyan/Downloads/homework1/deliverables/test_features.txt','wb')
    keys = patient_features_test.keys()
    for k in keys:
        f_k = sorted(patient_features_test[k],key=lambda tup: tup[0])
        l =  utils.bag_to_svmlight(f_k) +" "+"\n"
        l_id = str(k).replace('.0',"") +" " +l
        deliverable.write(bytes((l_id),'UTF-8'))
    

    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''

def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
    '''
    X = X_train
    Y = Y_train
    iterNo = 5
    test_percent = 0.2
    for m in np.arange(3,8,1):
        for n in np.arange(400,800,50):
            cv = ShuffleSplit(n_splits=iterNo, test_size = test_percent, random_state=RANDOM_STATE)
            accuracy = 0
            AUC = 0
            for train, test in cv.split(X, Y):
                #print (train, X.iloc[train,:],Y.iloc[train,:])
                dt = DecisionTreeRegressor(max_depth=m)
                abr = AdaBoostRegressor(dt,n_estimators=n)
                abr.fit(X.iloc[train,:],Y.iloc[train,:])
                y_pred = abr.predict(X.iloc[test,:])
                for i in range(len(y_pred)):
                    if y_pred[i]>=0.5:
                        y_pred[i]=1
                    else:
                        y_pred[i]=0
                #print(y_pred)
                accuracy += accuracy_score(Y.iloc[test,:],y_pred)
                fpr, tpr, thresholds = roc_curve(Y.iloc[test,:], y_pred)
                AUC += auc(fpr, tpr)
            accuracy = accuracy/iterNo
            AUC = AUC/iterNo
            print(m,n, accuracy, AUC)
    '''
    dt = DecisionTreeRegressor(max_depth=5)
    abr = AdaBoostRegressor(dt,n_estimators= 600)
    abr.fit(X_train,Y_train)
    y_pred = abr.predict(X_test)
    
    for i in range(len(y_pred)):
        if y_pred[i]>=0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0
    
    return y_pred



def main():
    filepath = "C:/Users/yyan/Downloads/homework1/data/"
    X_train, Y_train, X_test = my_features(filepath+"train/", filepath+"test/")
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
    my_prediction = pd.DataFrame(Y_pred, columns=['label'],index=X_test.index).reset_index()
    my_prediction.columns=['patient_id','label']
    my_prediction.to_excel('C:/Users/yyan/Downloads/homework1/deliverables/my_predictions.xlsx',index=False)
    #my_prediction.to_csv('C:/Users/yyan/Downloads/homework1/deliverables/my_predictions.csv')
    utils.generate_submission("C:/Users/yyan/Downloads/homework1/deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	