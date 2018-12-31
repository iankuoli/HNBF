function [log_posteriori] = Evaluate_LogPosterior_Poisson(vec_obsrv, n_scale, vec_theta, vec_lambda)
    %
    % The criterion is for paper: Coupled Compound Poisson Factorization
    % ref: Basbug, M. E., & Engelhardt, B. E. (2017). Coupled Compound Poisson Factorization. arXiv preprint arXiv:1701.02058.
    %
    
    % matProb_ZTP: dim( length(vec_lambda) , n_scale )
    matProb_ZTP = bsxfun(@times, bsxfun(@times, bsxfun(@power, vec_lambda, 1:n_scale), exp(-vec_lambda)), 1./gamma(1:n_scale));
    
    % matProb_Compound: dim( length(vec_lambda) , n_scale )
    phi_n = 1:n_scale;
    matProb_Compound = bsxfun(@times, bsxfun(@times, bsxfun(@power, vec_theta * phi_n, vec_obsrv), exp(-phi_n), 1./gamma(vec_obsrv));
    
    log_posteriori = log(sum(matProb_Compound .* matProb_ZTP, 2));
end