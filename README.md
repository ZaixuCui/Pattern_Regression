# Pattern_Regression_Python

This is the code of our NeuroImage paper (https://www.sciencedirect.com/science/article/pii/S1053811918305081) about comparison of different pattern regression algorithms (i.e., OLS, Ridge, LASSO, Elastic-Net, SVR, RVR) and evaluation the impact of sample size on the prediction performance. 

The scikit-learn library (version: 0.16.1) was used to implement OLS regression, LASSO regression, ridge regression and elastic-net regression (http://scikit-learn.org/) (Pedregosa et al., 2011), the LIBSVM function in MATLAB was used to implement LSVR (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Chang and Lin, 2011), the PRoNTo toolbox (http://www.mlnl.cs.ucl.ac.uk/pronto/) was used to implement RVR (Schrouff et al., 2013). 

C parameter of LSVR is the coefficient of training error, and λ parameter of LASSO/ridge/elastic-net regression is the coefficient of the regularization term, which contrasts one another. Therefore, C was chosen from among 16 values [2-5, 2-,4, …, 29, 210] (Hsu et al., 2003), and accordingly, λ was chosen from among 16 values [2-10, 2-,9, …, 24, 25]. Specifically, for elastic-net regression, we applied a grid search in which λ was chosen from among the 16 values above, and α was chosen from among 11 values, i.e., [0, 0.1, …, 0.9, 1].

Note: Codes for RVR will not work well in Matlab higher than 2012 version.
