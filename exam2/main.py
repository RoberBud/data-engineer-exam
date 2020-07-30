
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
    test_data.index =  min_index
   
    test_data.rename(columns={'datetime':'temp'})
    
    new = test_data['EventId'].groupby('datetime').agg(list)
  
    new.to_csv(f'./result/{file}.csv')
    
base_reader('train')
base_reader('test')








