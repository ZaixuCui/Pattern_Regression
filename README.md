# Pattern_Regression_Python

This is the code of our NeuroImage paper (https://www.sciencedirect.com/science/article/pii/S1053811918305081) about the comparison of different pattern regression algorithms (i.e., OLS, Ridge, LASSO, Elastic-Net, SVR, RVR) and the impact of sample size on the prediction performance. 

If you would like to do individualized behavior prediction in your work. It's better to try codes in https://github.com/ZaixuCui/Pattern_Regression_Clean. The codes here are more specific to this study. 

Citing our related paper will be greatly appreciated if you use these codes.
<br>&emsp; ```Zaixu Cui, Gaolang Gong, The effect of machine learning regression algorithms and sample size on individualized behavioral prediction with functional connectivity features, NeuroImage, Volume 178, Pages 622-637```
<br>&emsp; ```Zaixu Cui, et al., Individualized Prediction of Reading Comprehension Ability Using Gray Matter Volume, Cerebral Cortex, Volume 28, Issue 5, 1 May 2018, Pages 1656–1672, https://doi.org/10.1093/cercor/bhx061```
<br>&emsp; ```Zaixu Cui, et al., Individual variation in functional topography of association networks in youth. (2020) Neuron, 106(2): 340-53.```
<br>&emsp; ```Zaixu Cui, et al., Optimization of energy state transition trajectory supports the development of executive function during youth. (2020) eLife. 9:e53060. ```
<br>&emsp; ```Zaixu Cui, et al., 2016. Disrupted white matter connectivity underlying developmental dyslexia: A machine learning approach. Hum Brain Mapp 37, 1443-1458.```

The scikit-learn library (version: 0.16.1) was used to implement OLS regression, LASSO regression, ridge regression and elastic-net regression (http://scikit-learn.org/) (Pedregosa et al., 2011), the LIBSVM function in MATLAB was used to implement LSVR (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Chang and Lin, 2011), the PRoNTo toolbox (http://www.mlnl.cs.ucl.ac.uk/pronto/) was used to implement RVR (Schrouff et al., 2013). 

C parameter of LSVR is the coefficient of training error, and λ parameter of LASSO/ridge/elastic-net regression is the coefficient of the regularization term, which contrasts one another. Therefore, C was chosen from among 16 values [2-5, 2-,4, …, 29, 210] (Hsu et al., 2003), and accordingly, λ was chosen from among 16 values [2-10, 2-,9, …, 24, 25]. Specifically, for elastic-net regression, we applied a grid search in which λ was chosen from among the 16 values above, and α was chosen from among 11 values, i.e., [0, 0.1, …, 0.9, 1].

Note: Codes for RVR will not work well in Matlab higher than 2012 version.
