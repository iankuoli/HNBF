function newPRPF(usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global PRPF_matTheta        % dim(M, K): latent document-topic intensities
    global PRPF_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
    global PRPF_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

    global PRPF_matBeta         % dim(N, K): latent word-topic intensities
    global PRPF_matBeta_Shp     % dim(M, K): varational param of matBeta (shape)
    global PRPF_matBeta_Rte     % dim(M, K): varational param of matBeta (rate)

    global PRPF_matEpsilon      % dim(N, K): latent word-topic intensities
    global PRPF_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
    global PRPF_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

    global PRPF_matEta          % dim(N, K): latent word-topic intensities
    global PRPF_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
    global PRPF_matEta_Rte      % dim(N, 1): varational param of matEta (rate)
    
    global PRPF_prior
    
    global matX_train
    global matX_predict
    
    
    %% Intialization        
    ini_scale = PRPF_prior(1)/100;
    
    a = PRPF_prior(1);
    b = PRPF_prior(2);
    c = PRPF_prior(3);
    d = PRPF_prior(4);
    e = PRPF_prior(5);
    f = PRPF_prior(6);
    
    PRPF_matEpsilon_Shp = ini_scale * rand(M, 1) + b;
    PRPF_matEpsilon_Rte = ini_scale * rand(M, 1) + c;
    PRPF_matEpsilon = PRPF_matEpsilon_Shp ./ PRPF_matEpsilon_Rte;

    PRPF_matEta_Shp = ini_scale * rand(N, 1) + e;
    PRPF_matEta_Rte = ini_scale * rand(N, 1) + f;
    PRPF_matEta = PRPF_matEta_Shp ./ PRPF_matEta_Rte;

    PRPF_matBeta_Shp = ini_scale * rand(N, K) + d;
    PRPF_matBeta_Rte = bsxfun(@plus, ini_scale * rand(N, K), PRPF_matEta);
    PRPF_matBeta = PRPF_matBeta_Shp ./ PRPF_matBeta_Rte;
    PRPF_matBeta_Shp(itm_zeros, :) = 0;
    PRPF_matBeta_Rte(itm_zeros, :) = 0;
    PRPF_matBeta(itm_zeros, :) = 0;

    PRPF_matTheta_Shp = ini_scale * rand(M, K) + a;
    PRPF_matTheta_Rte = bsxfun(@plus, ini_scale * rand(M, K), PRPF_matEpsilon);
    PRPF_matTheta = PRPF_matTheta_Shp ./ PRPF_matTheta_Rte;
    PRPF_matTheta_Shp(usr_zeros,:) = 0;
    PRPF_matTheta_Rte(usr_zeros,:) = 0;
    PRPF_matTheta(usr_zeros,:) = 0;
    
    [is, js, vs] = find(matX_train);
    vs_tilde = sum(PRPF_matTheta(is,:) .* PRPF_matBeta(js,:), 2);
    matX_predict = sparse(is, js, vs_tilde, M, N);
    
end