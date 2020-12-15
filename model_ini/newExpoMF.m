function newExpoMF(ini_bias, init_delta, usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
        
    global matTheta        % dim(M, K): user factors
    global matBeta         % dim(N, K): item factors
    global vecMu           % dim(N, 1): exposure priors
    
    
    %% Intialization
    scale = sqrt(ini_bias / K);
    
    matTheta = init_delta * rand(M,K) + scale;
    matTheta(usr_zeros,:) = 0;

    matBeta = init_delta * rand(N,K) + scale;
    matBeta(itm_zeros, :) = 0;
    
    vecMu = init_delta * rand(N,1);
end