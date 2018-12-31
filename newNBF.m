function newNBF(init_delta, usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX_train
    
    global G_matTheta        % dim(M, K): latent document-topic intensities
    global G_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
    global G_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

    global G_matBeta         % dim(N, K): latent word-topic intensities
    global G_matBeta_Shp     % dim(N, K): varational param of matBeta (shape)
    global G_matBeta_Rte     % dim(N, K): varational param of matBeta (rate)
    
    global G_matEpsilon      % dim(M, 1): latent word-topic intensities
    global G_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
    global G_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

    global G_matEta          % dim(N, 1): latent word-topic intensities
    global G_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
    global G_matEta_Rte      % dim(N, 1): varational param of matEta (rate)

    global matD
    
    global G_prior
    
    
    %% Intialization
    [is_X_train, js_X_train, vs_X_train] = find(matX_train); 
    
    a = G_prior(1);
    b = G_prior(2);
    c = G_prior(3);
    d = G_prior(4);
    e = G_prior(5);
    f = G_prior(6);
    alpha_D = G_prior(7);
    
    G_matEpsilon_Shp = init_delta * rand(M, 1) + b;
    G_matEpsilon_Rte = init_delta * rand(M, 1) + c;
    G_matEpsilon = G_matEpsilon_Shp ./ G_matEpsilon_Rte;

    G_matEta_Shp = init_delta * rand(N, 1) + e;
    G_matEta_Rte = init_delta * rand(N, 1) + f;
    G_matEta = G_matEta_Shp ./ G_matEta_Rte;

    G_matBeta_Shp = bsxfun(@plus, init_delta * rand(N, K), a);
    G_matBeta_Rte = bsxfun(@plus, init_delta * rand(N, K), G_matEta);
    G_matBeta = G_matBeta_Shp ./ G_matBeta_Rte;
    G_matBeta_Shp(itm_zeros, :) = 0;
    G_matBeta_Rte(itm_zeros, :) = 0;
    G_matBeta(itm_zeros, :) = 0;

    G_matTheta_Shp = bsxfun(@plus, init_delta * rand(M, K), d);
    G_matTheta_Rte = bsxfun(@plus, init_delta * rand(M, K), G_matEpsilon);
    G_matTheta = G_matTheta_Shp ./ G_matTheta_Rte;
    G_matTheta_Shp(usr_zeros,:) = 0;
    G_matTheta_Rte(usr_zeros,:) = 0;
    G_matTheta(usr_zeros,:) = 0;
    
    matD = ones(M, N);
end