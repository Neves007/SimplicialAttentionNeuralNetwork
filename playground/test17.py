'''
绘制混淆矩阵
'''


import matplotlib.pyplot as plt
import seaborn as sns
glue = sns.load_dataset("glue",data_home="seaborn-data-master/").pivot(index="Model", columns="Task", values="Score")
sns.heatmap(glue, annot=True)
plt.show()