import os 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold

def kFold(df):
    
  df['kfold'] = -1
  df = df.reset_index(drop=True)
  y = df[1]
  kf = StratifiedKFold(n_splits=5)
  for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f

  return df

def get_training_dataset(path):
    """
    the expected directory structure

            +-- train
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
            +-- val
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
    """
    print("generating training dataset")

    uniques = ["NILM" , "ASC-US" , "ASC-H" , "LSIL" , "HSIL", "SCC"]
    dirs = ["train" , "val"]
    data = []

    for dir in dirs :
        for unique in uniques:
            directory = path + "/" + dir + "/" + unique

            for filename in os.listdir(directory):
                paths = directory + "/" + filename
                data.append([paths, unique])

    df = pd.DataFrame(data, columns = ["path", "class"])

    return df

def get_testing_dataset(path):
    '''
    the expected directory structure

            +-- test
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
    '''
    print("generating testing dataset")

    uniques = ["NILM" , "ASC-US" , "ASC-H" , "LSIL" , "HSIL", "SCC"]
    dirs = ["test"]
    data = []

    for dir in dirs:
        for unique in uniques:
            directory = os.path.join(path, dir, unique)

            for filename in os.listdir(directory):
                paths = os.path.join(directory, filename)
                data.append([paths, unique])
    
    test_df = pd.DataFrame(data, columns = ["path", "class"])

    return test_df