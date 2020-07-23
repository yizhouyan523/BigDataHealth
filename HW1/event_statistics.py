import time
import pandas as pd
import numpy as np
import datetime
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    df = events.set_index('patient_id').join(mortality.set_index('patient_id'),lsuffix='',rsuffix='_d')
    deaths = df[df['label']==1]
    alive = df[df['label']!=1]
    avg_dead_event_count = deaths.groupby('patient_id')['event_id'].count().mean()
    max_dead_event_count = deaths.groupby('patient_id')['event_id'].count().max()
    min_dead_event_count = deaths.groupby('patient_id')['event_id'].count().min()
    avg_alive_event_count = alive.groupby('patient_id')['event_id'].count().mean()
    max_alive_event_count = alive.groupby('patient_id')['event_id'].count().max()
    min_alive_event_count = alive.groupby('patient_id')['event_id'].count().min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    df = events.set_index('patient_id').join(mortality.set_index('patient_id'),lsuffix='',rsuffix='_d')
    df = df[df['event_id'].str.startswith(('DIAG','LAB','DRUG'))==True]
    deaths = df[df['label']==1]
    alive = df[df['label']!=1]
    avg_dead_encounter_count = deaths.groupby('patient_id').timestamp.nunique().mean()
    max_dead_encounter_count = deaths.groupby('patient_id').timestamp.nunique().max()
    min_dead_encounter_count = deaths.groupby('patient_id').timestamp.nunique().min()
    avg_alive_encounter_count = alive.groupby('patient_id').timestamp.nunique().mean()
    max_alive_encounter_count = alive.groupby('patient_id').timestamp.nunique().max()
    min_alive_encounter_count = alive.groupby('patient_id').timestamp.nunique().min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    df = events.set_index('patient_id').join(mortality.set_index('patient_id'),lsuffix='',rsuffix='_d')
    deaths = df[df['label']==1]
    deaths_delta = deaths.groupby('patient_id').timestamp.max().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')) - deaths.groupby('patient_id').timestamp.min().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')) 
    deaths_delta = deaths_delta.apply(lambda x : x.days)
    alive = df[df['label']!=1]
    alive_delta = alive.groupby('patient_id').timestamp.max().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')) - alive.groupby('patient_id').timestamp.min().apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')) 
    alive_delta = alive_delta.apply(lambda x : x.days)

    avg_dead_rec_len = deaths_delta.mean()
    max_dead_rec_len = deaths_delta.max()
    min_dead_rec_len = deaths_delta.min()
    avg_alive_rec_len = alive_delta.mean()
    max_alive_rec_len = alive_delta.max()
    min_alive_rec_len = alive_delta.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = 'C:/Users/yyan/Downloads/homework1/data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()