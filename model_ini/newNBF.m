function newNBF(init_delta, usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX_train
    
    global matTheta        % dim(M, K): latent document-topic intensities
    global matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
    global matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

    global matBeta         % dim(N, K): latent word-topic intensities
    global matBeta_Shp     % dim(N, K): varational param of matBeta (shape)
    global matBeta_Rte     % dim(N, K): varational param of matBeta (rate)
    
    global matEpsilon      % dim(M, 1): latent word-topic intensities
    global matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
    global matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

    global matEta          % dim(N, 1): latent word-topic intensities
    global matEta_Shp      % dim(N, 1): varational param of matEta (shape)
    global matEta_Rte      % dim(N, 1): varational param of matEta (rate)

    global matD
    
    global prior
    
    
    %% Intialization
    [is_X_train, js_X_train, vs_X_train] = find(matX_train); 
    
    a = prior(1);
    b = prior(2);
    c = prior(3);
    d = prior(4);
    e = prior(5);
    f = prior(6);
    alpha_D = prior(7);
    
    matEpsilon_Shp = init_delta * rand(M, 1) + b;
    matEpsilon_Rte = init_delta * rand(M, 1) + c;
    matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

    matEta_Shp = init_delta * rand(N, 1) + e;
    matEta_Rte = init_delta * rand(N, 1) + f;
    matEta = matEta_Shp ./ matEta_Rte;

    matBeta_Shp = bsxfun(@plus, init_delta * rand(N, K), a);
    matBeta_Rte = bsxfun(@plus, init_delta * rand(N, K), matEta);
    matBeta = matBeta_Shp ./ matBeta_Rte;
    matBeta_Shp(itm_zeros, :) = 0;
    matBeta_Rte(itm_zeros, :) = 0;
    matBeta(itm_zeros, :) = 0;

    matTheta_Shp = bsxfun(@plus, init_delta * rand(M, K), d);
    matTheta_Rte = bsxfun(@plus, init_delta * rand(M, K), matEpsilon);
    matTheta = matTheta_Shp ./ matTheta_Rte;
    matTheta_Shp(usr_zeros,:) = 0;
    matTheta_Rte(usr_zeros,:) = 0;
    matTheta(usr_zeros,:) = 0;
    
    matD = ones(M, N);
end