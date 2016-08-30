import numpy as np #numerical function
import datetime
import multiprocessing as mp

cores= mp.cpu_count() #processes raw data in batches (required because data>ram)
loding = True

## parameters
long_gap_length=7*60
gap_window=5*60 #ie long gaps are 7 hours<gap<11 hours
shor_gap_length=15 #analysis extremely robust to changes in this parameter

#guesses at what bounds waking time across the population
early_rising=5
late_rising=12

#guesses at what bounds bedtime across the population
early_daytim=early_rising+12
late_daytim=late_rising+12


def add_TotalPlays(data_frame):
    # print("add_Total")
    ## add players total number of attempts to df
    data_frame['total_plays'] = len(data_frame)
    return data_frame

def add_GapType(df):
    # print("add_GapType")
    df['localtime']=np.around((df['ga:hour']+df['ga:longitude']*24/360)%24 , decimals=1)
    df['long_gap']=(df['diff_time'].any()>long_gap_length) & (df['diff_time'].any()<(long_gap_length+gap_window))
    df['shor_gap']=df['diff_time']<shor_gap_length
    df['sleepgap']=df['long_gap'] & ((df['localtime'].any()>early_rising) & (df['localtime'].any()<late_rising))
    return df

def add_GapCategory(df):
    # print("add_GapCategory")
    '''categorise people by their gapness'''
    '''here define range over which we define taking gaps'''
    #dfp=df[6:8] #tight range, on game 7
    dfp=df[1:15] #any of first 15 games
    
    if len(dfp.shor_gap)==sum(dfp.shor_gap):
        gaptype=1
    elif (sum(dfp.long_gap)==1) & (sum(dfp.sleepgap)==0):
        gaptype=2
    elif (sum(dfp.long_gap)==1) & (sum(dfp.sleepgap)==1):
        gaptype=3
    else:
        gaptype=4
    
    df['gaptype']=gaptype

    return df

def add_DiffTime(df):
    # print("add_Diff")
    for index, row in df.iterrows():
        '''
        if not index%10:
            print("{0}: {1}%".format(str(queue), round(index/len(new_df)*100)))
        
        count = count+1
        '''
        ## combine day, hour and minute into datetime object (ie absolute time)
        str_date_time = "{0}:{1}:{2}".format(str(int(row['ga:date'])), str(int(row['ga:hour'])), str(int(row['ga:minute'])))
        str_to_time = datetime.datetime.strptime(str_date_time, "%Y%m%d:%H:%M")
        
        df.loc[index, 'comb_time'] = str_to_time
        
        if df.loc[index,'ga:eventLabel']==1:
            df.loc[index, 'diff_time'] = np.NaN
        else:
            try:
                diff_time = (df.loc[index,'comb_time'].to_datetime() - df.loc[index-1,'comb_time'].to_datetime()).total_seconds()/60
                df.loc[index, 'diff_time'] = diff_time
            except:
                df.loc[index, 'diff_time'] = np.NaN
    return df

def check_Discon(df, name_vec):
    # print("check_Discon")
    count = 0
    groups = df.groupby('ga:eventAction')
    
    for name, group in groups:
        df_len = len(group)
        eventLabel = np.ones(df_len)
        eventLabel[:] = group['ga:eventLabel']
        diff_AlOm = eventLabel[df_len-1]-eventLabel[0]+1
        
        if not diff_AlOm == df_len:
            name_vec.append(name)
        count = count+1

def filter_time(df):
    df=df[~(df['ga:latitude']==0)]
    '''remove data with negative time differences'''
    '''cannot work out why some time differences are <0, but have verified that >0 values are correct'''
    df=df[~(df.diff_time<0)]
    
    return df

def rm_GaInCols(data_frame): # remove ga:
    # print("rm_GaInCols")
    cols = np.array(data_frame.columns)
    count = 0
    for elmt in cols:
        if 'ga:' in elmt:
            cols[count] = elmt.strip('ga:')
        count = count + 1
    return cols

def data_PostProcess(df, err_plys):
    # print("data_PostProcess")
    df.sort_values(by=['ga:eventAction', 'ga:eventLabel'], inplace=True) # mac
    #df.sort(columns=['ga:eventAction', 'ga:eventLabel'],inplace=True) # school computer
    df = df.reset_index()
    err_plys = err_plys.reset_index()
    
    df = df.drop('index', axis=1)
    df.columns = rm_GaInCols(df) # remove ga:
    
    err_plys = err_plys.drop('index', axis=1)
    err_plys.columns = ['player ID']
    
    return df, err_plys