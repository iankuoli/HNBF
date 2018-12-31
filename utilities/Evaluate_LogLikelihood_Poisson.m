function [ret] = Evaluate_LogLikelihood_Poisson(vec_obsrv, vec_pred)
    
    % poisson(k | lambda) = lambda^K * exp(-lambda) ./ gamma(k+1);
    
    ret = vec_obsrv .* log(vec_pred) - vec_pred - gammaln(vec_obsrv+1);
    
end