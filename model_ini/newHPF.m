function newHPF(init_delta, usr_zeros, itm_zeros, prior)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global HPF_matTheta        % dim(M, K): latent document-topic intensities
    global HPF_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
    global HPF_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

    global HPF_matBeta         % dim(N, K): latent word-topic intensities
    global HPF_matBeta_Shp     % dim(M, K): varational param of matBeta (shape)
    global HPF_matBeta_Rte     % dim(M, K): varational param of matBeta (rate)

    global HPF_matEpsilon      % dim(N, K): latent word-topic intensities
    global HPF_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
    global HPF_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

    global HPF_matEta          % dim(N, K): latent word-topic intensities
    global HPF_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
    global HPF_matEta_Rte      % dim(N, 1): varational param of matEta (rate)
    
    global HPF_prior
    
    
    %% Intialization
    HPF_prior = prior;
    a = HPF_prior(1);
    b = HPF_prior(2);
    c = HPF_prior(3);
    d = HPF_prior(4);
    e = HPF_prior(5);
    f = HPF_prior(6);
    
    HPF_matEpsilon_Shp = init_delta * rand(M, 1) + b;
    HPF_matEpsilon_Rte = init_delta * rand(M, 1) + c;
    HPF_matEpsilon = HPF_matEpsilon_Shp ./ HPF_matEpsilon_Rte;

    HPF_matEta_Shp = init_delta * rand(N, 1) + e;
    HPF_matEta_Rte = init_delta * rand(N, 1) + f;
    HPF_matEta = HPF_matEta_Shp ./ HPF_matEta_Rte;

    HPF_matBeta_Shp = init_delta * rand(N, K) + d;
    HPF_matBeta_Rte = bsxfun(@plus, init_delta * rand(N, K), HPF_matEta);
    HPF_matBeta = HPF_matBeta_Shp ./ HPF_matBeta_Rte;
    HPF_matBeta_Shp(itm_zeros, :) = 0;
    HPF_matBeta_Rte(itm_zeros, :) = 0;
    HPF_matBeta(itm_zeros, :) = 0;

    HPF_matTheta_Shp = init_delta * rand(M, K) + a;
    HPF_matTheta_Rte = bsxfun(@plus, init_delta * rand(M, K), HPF_matEpsilon);
    HPF_matTheta = HPF_matTheta_Shp ./ HPF_matTheta_Rte;
    HPF_matTheta_Shp(usr_zeros,:) = 0;
    HPF_matTheta_Rte(usr_zeros,:) = 0;
    HPF_matTheta(usr_zeros,:) = 0;
    
    
end