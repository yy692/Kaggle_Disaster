import pandas as pd
import numpy as np


def gen_validation(fname, ratio=0.8, random_state=1):
    df = pd.read_csv(fname)

    length = df.shape[0]
    len_val = int(length*0.8)

    df_val = df.sample(n=len_val, random_state=random_state)
    df_test = df.drop(df_val.index)    

    df_val.to_csv('validation.csv', index=False) 
    df_test.to_csv('pre_test.csv', index=False) 

    print(pd.read_csv('validation.csv').shape)
    print(pd.read_csv('pre_test.csv').shape)

if __name__=='__main__':
    gen_validation('train.csv')
