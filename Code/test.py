# -*- coding: utf-8 -*-
import pandas as pd #data analysis
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
import DC_Pickle as dcp
import DC_filter as dcf
t0 = time.clock() # initial time
cores= mp.cpu_count() #processes raw data in batches (required because data>ram)
loding = True

def process_df(queue, file_name):
    err_players = []
    
    df = dcp.open_Pickle(file_name)
    df = df.drop('Unnamed: 0', axis=1)
    
    new_df = df.groupby('ga:eventAction').apply(dcf.add_TotalPlays)
    new_df = new_df[new_df['total_plays']>14]
    new_df = new_df[new_df['ga:eventLabel']<301]
    dcf.check_Discon(new_df, err_players)
                  
    new_df = dcf.add_DiffTime(new_df)
    new_df = dcf.add_GapType(new_df)
    #df = filter_time(df) # 이건 안하는게 좋겠다. attemps를 지우니까.
    new_df=new_df.groupby('ga:eventAction').apply(dcf.add_GapCategory)
                  
    queue.put([new_df, err_players])

## Execute! ##
print('\n -  Data Loading...')

## start multiprocessing
for i in range(cores):
    set_filename = "../../data/pickles/multiprocessing_test/process{0}.pickle".format(i)
    set_pros_name = "core{0}".format(i)
    set_q_name = "q{0}".format(i)

    globals()[set_q_name] = Queue()
    globals()[set_pros_name] = Process(target=process_df, args=(eval(set_q_name), set_filename,))
    #globals()[set_pros_name] = Process(target=process_df, args=(eval(set_q_name), set_filename, eval(error_name),))

    eval(set_pros_name).start()
    print('core{0} start'.format(i))
    time.sleep(1)

print('\n -  Data filtering...')

## get returned value from multiprocessing
df = pd.DataFrame()
err = pd.DataFrame()
for i in range(cores):
    set_q_name = "q{0}".format(i) # queue name
    set_pros_name = "core{}".format(i) # process name
    set_df_name = "df{0}".format(i) # data frame name of each process
    error_name = "error_players{0}".format(i) # player name with error attempts
    
    locals()[set_df_name], locals()[error_name] = eval(set_q_name).get() # get from queue

    df = df.append(eval(set_df_name)) # concat each dfs
    err = err.append(eval(error_name)) # np.hstack((err_plys, eval(error_name))) # concat each error players
    #df = df.drop('level_0', axis=1)
    
    eval(set_q_name).close()
    eval(set_pros_name).join()
    print('core{0} close'.format(i))
    time.sleep(1)

[df, err] = dcf.data_PostProcess(df, err)

print("\n - saving processed data...\n {0} plays, {1}players".format(len(df), len(df.groupby('eventAction'))))

df.to_csv('../../data/filtered_data_test.csv')
err.to_csv('../../data/error_players_test.csv')
print("\n - process terminal.")
print("Run time: ", time.clock()-t0)