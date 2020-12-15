function newCCPF_HPF(init_delta, usr_zeros, itm_zeros)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
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
    
    global G_vecMu           % dim(M, 1): approximate matD
    global G_vecMu_Shp       % dim(M, 1): approximate matD
    global G_vecMu_Rte       % dim(M, 1): approximate matD
    
    global G_matGamma        % dim(M, 1): latent word-topic intensities
    global G_matGamma_Shp    % dim(M, 1): varational param of matEpsilonX (shape)
    global G_matGamma_Rte    % dim(M, 1): varational param of matEpsilonX (rate)

    global G_vecPi           % dim(N, 1): approximate matD
    global G_vecPi_Shp       % dim(N, 1): approximate matD
    global G_vecPi_Rte       % dim(N, 1): approximate matD
    
    global G_matDelta        % dim(N, 1): latent word-topic intensities
    global G_matDelta_Shp    % dim(N, 1): varational param of matEtaX (shape)
    global G_matDelta_Rte    % dim(N, 1): varational param of matEtaX (rate)
    
    global G_prior
    
    
    %% Intialization
    
    a = G_prior(1);
    b = G_prior(2);
    c = G_prior(3);
    d = G_prior(4);
    e = G_prior(5);
    f = G_prior(6);
    a_gamma = G_prior(7);
    a_mu = G_prior(8);
    b_mu = G_prior(9);
    a_delta = G_prior(10);
    a_pi = G_prior(11);
    b_pi = G_prior(12);
    
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
    
    G_matGamma_Shp = bsxfun(@plus, init_delta * rand(M, K), a_gamma);
    G_matGamma_Rte = bsxfun(@plus, init_delta * rand(M, K), a_mu / b_mu);
    G_matGamma = G_matGamma_Shp ./ G_matGamma_Rte;
    G_matGamma_Shp(usr_zeros,:) = 0;
    G_matGamma_Rte(usr_zeros,:) = 0;
    G_matGamma(usr_zeros,:) = 0;
    
    G_matDelta_Shp = bsxfun(@plus, init_delta * rand(N, K), a_delta);
    G_matDelta_Rte = bsxfun(@plus, init_delta * rand(N, K), a_pi / b_pi);
    G_matDelta = G_matDelta_Shp ./ G_matDelta_Rte;
    G_matDelta_Shp(itm_zeros, :) = 0;
    G_matDelta_Rte(itm_zeros, :) = 0;
    G_matDelta(itm_zeros, :) = 0;

    G_vecMu_Shp = a_mu + init_delta * rand(M, 1);
    G_vecMu_Rte = b_mu + init_delta * rand(M, 1);
    G_vecMu = G_vecMu_Shp ./ G_vecMu_Rte;
    
    G_vecPi_Shp = a_pi + init_delta * rand(N, 1);
    G_vecPi_Rte = b_pi + init_delta * rand(N, 1);
    G_vecPi = G_vecPi_Shp ./ G_vecPi_Rte;
end