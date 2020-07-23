import utils
import pandas as pd
import datetime
import numpy as np
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    df = events.set_index('patient_id').join(mortality.set_index('patient_id'),lsuffix='',rsuffix='_d')
    deaths = df[df['label']==1]
    alive = df[df['label']!=1]

    deaths_indx = deaths['timestamp_d'].groupby(by=['patient_id']).first().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')-datetime.timedelta(30)).to_frame()
    alive_indx = alive['timestamp'].groupby(by=['patient_id']).max().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')).to_frame()
    deaths_indx.columns=['indx_date']
    alive_indx.columns=['indx_date']
    
    indx_date = pd.concat([alive_indx,deaths_indx]).reset_index()
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv',columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    df = events.join(indx_date.set_index('patient_id'),on='patient_id',lsuffix='',rsuffix='_r')
    df['start_date'] = df['indx_date'].apply(lambda x: x-datetime.timedelta(2000))
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    df = df[(df['timestamp']>=df['start_date']) & (df['timestamp']<=df['indx_date'])]
    filtered_events = df[['patient_id','event_id','value']]
    
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', index=False)
    
    return filtered_events


def aggregate_events(filtered_events, mortality,feature_map, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return aggregated_events
    '''
    df = filtered_events.join(feature_map.set_index('event_id'),on='event_id').dropna(subset=['value'])
    sub_sum = df[df['event_id'].str.startswith(('DIAG','DRUG'))==True].groupby(by=['patient_id','idx']).sum()
    sub_count = df[df['event_id'].str.startswith(('LAB'))==True].groupby(by=['patient_id','idx']).count()
    sub_count = sub_count[['value']]

    columns=['patient_id', 'feature_id', 'feature_value']
    aggregated_events = pd.concat([sub_sum,sub_count]).reset_index()
    aggregated_events.columns = columns
    aggregated_events['feature_value'] = aggregated_events['feature_value']/aggregated_events.groupby(['feature_id'])['feature_value'].transform('max')

    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = 'C:/users/yyan/Downloads/homework1/deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    df = aggregated_events.join(mortality.set_index('patient_id'),on='patient_id',lsuffix='',rsuffix='_r')
    patient_features=df.set_index('patient_id')[['feature_id', 'feature_value']].T.apply(tuple).to_frame()
    patient_features.columns=['features']
    patient_features=patient_features.groupby(by=['patient_id'])['features'].apply(np.array)
    mortality = df.fillna(0).drop_duplicates().set_index('patient_id')['label'].to_dict()
    #print(patient_features)
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    keys = mortality.keys()
    for k in keys:
        f_k = sorted(patient_features[k],key=lambda tup: tup[0])
        l = str(mortality[k])  +" "+ utils.bag_to_svmlight(f_k) +" "+"\n"
        l_id = str(k).replace('.0',"") +" " +l
        deliverable1.write(bytes((l),'UTF-8'))
        deliverable2.write(bytes((l_id),'UTF-8'))
     

def main():
    train_path = 'C:/users/yyan/Downloads/homework1/data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, 'C:/users/yyan/Downloads/homework1/deliverables/features_svmlight.train', 'C:/users/yyan/Downloads/homework1/deliverables/features.train')

if __name__ == "__main__":
    main()