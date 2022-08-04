## 实验目的
这个实验是在对比gurobi和autodiff_abess在给定稀疏度下消耗的时间。

## 数据生成
信噪比为6，相关性为$\Sigma_{ij}=\rho^{|i-j|}, \rho=0.2$.

n,k,p三个参数固定两个，变动一个。

## 评价指标
由于稀疏度给定了，只需accuracy=TP/(TP+FN)即可表示混淆矩阵。
out of sample mse 测了几下发现数据没什么意义，因为两种算法性能很接近。
