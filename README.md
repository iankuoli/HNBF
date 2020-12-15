# Hierarchical Negative Binomial Factorization
>When exposed to an item in a recommender system, a user may consume it (known as success exposure) or neglect it (known as failure exposure). The recently proposed methods that consider both success and failure exposure merely regard failure exposure as a constant prior, thus being capable of neither modeling various user behavior nor adapting to overdispersed data. In this paper, we propose a novel model, hierarchical negative binomial factorization, which models data dispersion via a hierarchical Bayesian structure, thus alleviating the effect of data overdispersion to help with performance gain for recommendation. Moreover, we factorize the dispersion of zero entries approximately into two low-rank matrices, thus reducing the updating time linear to the number of nonzero entries. The experiment shows that the proposed model outperforms state-of-the-art Poisson-based methods merely with a slight loss of inference speed.

## Required library
- <a href="https://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection" target="_blank">Min/Max selection</a> 

## Usage
Add the included folders to path and run
```matlab
test_FastHNBF
```

## Experimental results on implicit count data
<img height=600 src="https://github.com/iankuoli/HNBF/blob/master/results_implicit.png?raw=true" />
