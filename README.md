# Hierarchical Negative Binomial Factorization
>In recommender systems, when being exposed to an item, a user may consume it (success exposure) or may not consume it (failure exposure). Most of the prior works on matrix factorization merely consider the former and omit the latter. In addition, classical methods which assume each observation over a Poisson cannot be feasible to overdispersed data. In this paper, we propose a novel model, hierarchical negative binomial factorization (HNBF), which models the perturbed dispersion by a hierarchical Bayesian structure rather than assigning a constant to the prior of the dispersion directly, thus alleviating the effect of data overdispersion. Moreover, we estimate the dispersion of zero entries approximately by utilizing matrix factorization, thus limiting the computational cost of updating per epoch linear to the number of nonzero entries. In the experiment, we show that the proposed method outperforms the state-of-the-art methods in terms of precision and recall on implicit data.

## Required library
- <a href="https://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection" target="_blank">Min/Max selection</a> 

## Usage
Add the included folders to path and run
```matlab
test_FastHNBF
```
