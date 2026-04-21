import pandas as pd 
import numpy as np 

# Mock df
df = pd.DataFrame({'a': [1], 'b': [2]})

print(df.head())
print(df.isnull())
print(df.describe())
print(df.head())          
print(df.tail())        
print(df.shape)
print(df.columns )        
print(df.info() )         
print(df.dtypes)
