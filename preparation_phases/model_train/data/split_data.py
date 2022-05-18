import pandas as pd

def split_data(target,frac=0.1):
    import random
    # open dataset
    df = pd.read_csv("./{}/{}.csv".format(target, target))  
    index=list(df.index)

    # random sample
    val_index=random.sample(index,int(frac*len(index))) 
    for val in val_index:
        index.remove(val)
    train_index=index

    # test dataset
    val_df=df.loc[val_index,:]
    val_df.to_csv("./{}/{}_val.csv".format(target, target), index=False, sep=',')
    
    # train dataset
    train_df=df.loc[train_index,:]
    train_df.to_csv(".//{}/{}_train.csv".format(target, target), index=False, sep=',')

if __name__ == "__main__":
    import sys
    split_data(sys.argv[1])