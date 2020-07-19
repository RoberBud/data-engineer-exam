
import os
import pandas as pd

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')

    pass


def base_reader(file):
    test_data = pd.read_csv(f'./data/{file}_need_aggregate.csv')
    test_data['datetime'] = pd.to_datetime(test_data['datetime'])
    test_data.index = pd.to_datetime(test_data['datetime'])
    min_index = test_data.index.to_period('T')
    test_data.set_index(min_index, inplace =True)
    
    drop = test_data[~test_data.index.duplicated()]
    drop = drop.index
    
    drop2 = drop.to_timestamp()
    column = [[] for i in range(len(drop2))]
    new = pd.Series(column, index=drop2)
    
    
    for index, row in test_data.iterrows():
        index_time = index.to_timestamp()
        res = new.loc[index_time]
        
        res.append(row[1])
        
        new.loc[index_time] = res 
        
   
    
    new.to_csv(f'./result/{file}.csv')
    
base_reader('train')
base_reader('test')








