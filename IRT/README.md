# run IRT

1.数据集放在跟IRT.py同一目录下并命名为rating.txt
2.数据集格式：
user序号	item序号	rating	time（以'\t'分开，user,item的序号都从0开始）

3.运行的命令
# python IRT.py M N alpha percentage

M: 学生数量
N：试题数量
alpha:  学习率
percentage:测试集的百分比

例如：python IRT.py 3217 411 0.001 0.3

