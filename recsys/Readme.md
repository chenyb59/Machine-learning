# Recommender System based on Probabilistics Matrix Factorization(PMF)

## Introduction to PMF
Matrix Factorization method maps every user and item into an abstract same space with k dimension, where k is always a low number to represent the number of latent factor.
The Probabilistic Matrix Factorization (PMF) raises a solution of how to factorize the extremely large rating matrix from a probabilistic view.\
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/WechatIMG15.png" width="150" height="50">

## Formulation inference
Suppose the original rating matrix,  satisfy the Normal Distribution as follows:
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/61550810911_.pic.jpg" width="420" height="65">

The priori probability of user feature matrix and item feature matrix are :\
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/71550810919_.pic.jpg" width="300" height="100">

According to Bayes Theorem, the posterior probability of these two matrix is : 
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/81550810930_.pic.jpg" width="700" height="150">

Take log operation to turn the multiplication into summation.
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/91550810940_.pic.jpg" width="750" height="60">

The U,V matrix which make this probability largest is the optimal solution. 
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/101550810951_.pic.jpg" width="400" height="70">\
   <img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/111550811114_.pic.jpg" width="120" height="45">\
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/121550811125_.pic.jpg" width="450" height="250">

Do similar operation to flatten the whole equation then we get :
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/131550811147_.pic_hd.jpg" width="600" height="120">

So the objective function is : \
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/141550811177_.pic.jpg" width="500" height="80">


## Implementation
- Local Version\
[SGD](https://github.com/chenyb59/Machine-learning/tree/master/recsys/sgd_version)\
[mini-batch](https://github.com/chenyb59/Machine-learning/tree/master/recsys/batch_version)\
U,V update rule : 
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/161550817151_.pic.jpg" width="400" height="100">

- Distributed Version\
[ALS](https://github.com/chenyb59/Machine-learning/tree/master/recsys/husky_version)\
update rule :
<img src="https://github.com/chenyb59/Machine-learning/blob/master/recsys/pic/171550817176_.pic.jpg" width="200" height="50">

Husky : an efficient distributed computing platform, developed by my project supervisors's lab, which adopts a master-worker architecture and support many different computational models like iterative fine-grained machine learn- ing algorithms, or coarse-grained batch processing model.
Intuitively, with same condition of hardware, husky should be faster than other distributed platforms like Spark and occupies less memory.

## Performance
- Distributed Version\
<img src="https://raw.githubusercontent.com/chenyb59/Machine-learning/master/recsys/pic/rmse.jpg" width="350" height="300"><img src="https://github.com/chenyb59/Machine-learning/blob/master/recsys/pic/rmse_100.jpg" width="350" height="300">

Compare with Spark.ALS.Recommendation\
Dataset：Netflix\
setting：λ=0.01\
total epoch number：5 on spark（迭代太多次会爆栈）/ 20 on husky\
total executor cores：20


size | metric | spark | Husky
---  | ---    | ---   | ---
0.7million rating（10MB）| Speed | 2.323s/epoch | 2s/epoch
0.7million rating（10MB）| Accuracy（RMSE）| 1.67 | 1.5
7million rating（100MB）| Speed | 3.7228s/epoch | 3.99s/epoch
7million rating（100MB）| Accuracy（RMSE）| 1.03 | 0.95
 
