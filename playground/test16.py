import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
sns.set_theme(style="whitegrid")

def buid_data():
    # 创建一个空字典来存储生成的数据
    data = []

    # 生成100个数据点
    for i in range(1000):
        # 为每个数据点生成随机数
        k1_value = random.randint(0, 21)  # 生成0到21之间的随机整数
        k2_value = random.randint(0, 10)  # 生成0到10之间的随机整数
        error_value = random.random()      # 生成0到1之间的随机浮点数

        # 将生成的数据存储到字典中，使用i来生成key
        data.append({'k1': k1_value, 'k2': k2_value, 'error': error_value})

    # 现在data包含了100个随机生成的数据点

    df = pd.DataFrame(data)
    return df
def pretr_dataframe():
    df = buid_data()
    df_group = df.groupby(['k1','k2'],as_index=False)
    df_agg = df_group.agg({'error': 'mean'})
    return df_agg

df = pretr_dataframe()
# dfpi = df_agg.pivot(index='k1',columns=['k2'],values='error')
g = sns.relplot(
    data=df,
    x='k1', y='k2', hue="error", size="error",
    palette="vlag",
    hue_norm=(-1, 1), 
    edgecolor=".7",
    height=7, 
    sizes=(50, 250), 
    size_norm=(-.2, .8),
)
g.set(xlabel="", ylabel="", aspect="equal")

plt.xticks(range(0, 22))  # 设置x轴刻度为0到21
plt.yticks(range(0, 11))  # 设置y轴刻度为0到10
g.despine(left=True, bottom=True)
plt.show()
