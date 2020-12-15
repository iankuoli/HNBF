function newBNMF(init_delta, usr_zeros, itm_zeros, num_Gibbs, num_chains)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global BNMF_matTheta
    global BNMF_matBeta
    global vec_sigmam
    global vec_nlp
    
    
    %% Intialization
    
    BNMF_matTheta = init_delta * rand(M, K);
    BNMF_matTheta(usr_zeros, :) = 0;
    
    BNMF_matBeta  = init_delta * rand(N, K); 
    BNMF_matBeta  = bsxfun(@times, BNMF_matBeta, 1 ./ sum(BNMF_matBeta,2));
    BNMF_matBeta(itm_zeros, :) = 0;
    
    vec_sigmam = zeros(num_Gibbs * num_chains, 1);
    vec_nlp    = zeros(num_Gibbs * num_chains, 1);
    
end