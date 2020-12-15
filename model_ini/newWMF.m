function newWMF(ini_bias, init_delta, usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
        
    global G_matTheta        % dim(M, K): user factors
    global G_matBeta         % dim(N, K): item factors
    
    
    %% Intialization
    scale = sqrt(ini_bias / K);
    
    G_matTheta = init_delta * rand(M,K) + scale;
    G_matTheta(usr_zeros,:) = 0;

    G_matBeta = init_delta * rand(N,K) + scale;
    G_matBeta(itm_zeros, :) = 0;
    
end