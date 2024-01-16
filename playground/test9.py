
'''
'DataFrame' object has no attribute 'append'. Did you mean: '_append'?

'''
import pandas as pd

df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]})
print(df)
new_row = pd.Series({'A': 50, 'B': 60})
df = df._append(new_row, ignore_index=True)
print(df)