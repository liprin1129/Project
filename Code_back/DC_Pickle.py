import numpy as np
import pandas as pd #data analysis
import os #file/directory operations
from six.moves import cPickle as pickle

def make_Pickle(set_data, set_filename, force = False):
    if os.path.exists(set_filename) and not force:
        # You may override by setting force=True.  
        print('%s already present - Skipping pickling.' % set_filename)
    else :
        try :
            with open(set_filename, 'wb') as f:
                pickle.dump(set_data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e :
            print("Unable to save data to", set_filename, ': ', e)
            return

def make_folders(dir_name):
    set_dir_name = dir_name

    if os.path.exists(set_dir_name):
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_dir_name)
    else:
        try:
            os.makedirs(set_dir_name)
        except Exception as e:
            print('Unable to make', set_dir_name, ':', e)
            return

def open_Pickle(pickle_path):
    try :
        with open(pickle_path, 'rb') as f:
            train = pickle.load(f)
            return train
    except Exception as e :
        print("Unable to save data to", pickle_path, ': ', e)
        return
            
def seperate_indi(groups, num_cores, dir_name):
    nFiles = round(len(groups)/num_cores)
    
    groups_dict = dict(list(groups))
    
    count = 0
    for i in range(num_cores):
        set_file_name = dir_name.format(i)
        
        if count < num_cores-1:
            gb = pd.concat(list(groups_dict.values())[nFiles*i:nFiles*(i+1)])
            make_Pickle(gb, set_file_name)
            print('inside')
        else:
            print('last!')
            gb = pd.concat(list(groups_dict.values())[nFiles*i:])
            make_Pickle(gb, set_file_name)
        count = count + 1
    print('finish!')
        
def seperate_ColData(df, col_name): # seperate columns into each matrix
    groups = df.groupby('eventAction') # grouping
    col_matrix = np.ones((301, len(groups)))*np.nan # make a matrix with player attempts size.

    count = 0
    for name, group in groups:
        idx = np.array(group['eventLabel'])-1
        col_matrix[idx, count] = group[col_name]
        count = count + 1
    return col_matrix
    
def colToPickle(df):
    set_folder_name = '../../data/pickles/seperate/'
    make_folders(set_folder_name)    
    
    for col_name in np.array(df.columns):
        if not (col_name == 'eventAction' or col_name == 'comb_time'): # conver to picke except 'eventAction' and 'comb_time'
            print(col_name)
            set_mat = col_name
            locals()[set_mat] = seperate_ColData(df, col_name)
            make_Pickle(eval(set_mat), '../../data/pickles/seperate/{0}.pickle'.format(col_name))
        elif col_name == 'eventAction':
            print(col_name)
            groups = df.groupby('eventAction')
            names = pd.DataFrame({'ID':groups.grouper.result_index.values})
            make_Pickle(names, '../../data/pickles/seperate/eventAction.pickle')
        elif col_name == 'comb_time':
            print('comb_time : seperation code has not been written')