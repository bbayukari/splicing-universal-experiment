数据存放格式为：

    表1：info 记录数据集信息
    id, n, b1, b2, ...

    表2：data 记录数据集
    id, y, x1, x2, ...


commit: 
fa120b0: experiment: logistic_1
```
n_list = [n*100+100 for n in range(10)]

for n in n_list:
    for _ in range(5):
        p = 500
        data = make_glm_data(
            n=n,
            p=p,
            k=10,
            rho=0.2,
            family="binomial",
            corr_type="exp",
            snr=10 * np.log10(6),
            standardize=True,
        )
```
实验效果不理想，abess_autodiff和omp结果相似，且omp快很多。下面加大实验难度，以区分出二者效果。


commit: 
a86ab49: experiment: logistic_2
```
n_list = np.arange(100,600,20)
p = 500
k = 20

for n in n_list:
    for _ in range(10):
        data = make_glm_data(
            n=n,
            p=p,
            k=k,
            rho=0.2,
            family="binomial",
            corr_type="exp",
            snr=10 * np.log10(6),
            standardize=True,
        )
```
实验显示abess_autodiff在小样本情况下略优于omp，当样本量上升，两者效果相当。速度上当然omp快得多。
此外，实验中再次发现autodiff_abess中优化大量失败，但同样，autodiff_abess效果仅稍逊于原版abess，差异并不大，结果相似度随着样本量增加从70%增加到100%。
