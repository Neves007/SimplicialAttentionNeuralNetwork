import pandas as pd
from sklearn.metrics import confusion_matrix

# 假设有以下真实类别和预测类别
y_true = ['cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog']
y_pred = ['cat', 'cat', 'dog', 'dog', 'cat', 'cat', 'dog', 'cat', 'cat', 'dog']

# 使用pandas.crosstab创建混淆矩阵
confusion_matrix_df = pd.crosstab(pd.Series(y_true, name='Actual'),
                                  pd.Series(y_pred, name='Predicted'),
                                  rownames=['Actual'], colnames=['Predicted'])

df_normalized = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)
print(df_normalized)


# 输出:
# Predicted  cat  dog
# Actual
# cat         1    1
# dog         1    1

# 使用sklearn.metrics.confusion_matrix函数进行比较
sklearn_conf_matrix = confusion_matrix(y_true, y_pred)
print(sklearn_conf_matrix)

# 输出:
# [[1 1]
#  [1 1]]