## 低阶准确率

低阶情况下准确率很准0.94左右



但是，该模型loss是用的交叉熵，他的逻辑的是只要判断下一步是I态那么TP[1]越接近1越好。

所以target-predict-TP看起来是这样

<img src="E:\_2_workplace\Pycharm\我的论文\高阶动力学学习\01Simple Graph Single Epidemic\笔记\typora-pic\image-20230414170148322.png" alt="image-20230414170148322" style="zoom:33%;" />

其实作者这里用的是比尔森相关系数。

我们在0.2来做这个事情。
