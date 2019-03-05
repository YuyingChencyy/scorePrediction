# run IRT

1.数据集放在跟IRT.py同一目录下，并命名为rating.txt
2.数据集格式：
user序号    item序号  rating time（以'\t'分开，user,item的序号都从0开始）

3.运行的命令
# python PMF.py M N dimension alpha lamda percentage
M: 学生数量
N：试题数量
dimension: 分解的隐向量维度
alpha:  学习率
lamda:  正则系数
percentage:测试集的百分比

例如：python PMF.py 3217 411 10 0.001 0.001 0.3
