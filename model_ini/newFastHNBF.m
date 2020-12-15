function newFastHNBF(init_delta, usr_zeros, itm_zeros)

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
    
    global vecMu           % dim(M, 1): approximate matD
    global vecMu_Shp       % dim(M, 1): approximate matD
    global vecMu_Rte       % dim(M, 1): approximate matD
    global matGamma        % dim(M, 1): latent word-topic intensities
    global matGamma_Shp    % dim(M, 1): varational param of matEpsilonX (shape)
    global matGamma_Rte    % dim(M, 1): varational param of matEpsilonX (rate)

    global vecPi           % dim(N, 1): approximate matD
    global vecPi_Shp       % dim(N, 1): approximate matD
    global vecPi_Rte       % dim(N, 1): approximate matD
    global matDelta        % dim(N, 1): latent word-topic intensities
    global matDelta_Shp    % dim(N, 1): varational param of matEtaX (shape)
    global matDelta_Rte    % dim(N, 1): varational param of matEtaX (rate)
    
    global vec_matR_ui_shp
    global vec_matR_ui_rte
    global vec_matR_ui
    global vec_matD_ui_shp
    global vec_matD_ui_rte
    global vec_matD_ui
    
    global prior
    
    
    %% Intialization
    [is_X_train, js_X_train, vs_X_train] = find(matX_train); 
    
    a = prior(1);
    b = prior(2);
    c = prior(3);
    d = prior(4);
    e = prior(5);
    f = prior(6);
    g_mu = prior(7);
    h_mu = prior(8);
    g_pi = prior(9);
    h_pi = prior(10);
    g_R = prior(11);
    h_R = prior(12);
    
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
    
    matGamma_Shp = bsxfun(@plus, init_delta*1000 * rand(M, K), h_mu);
    matGamma_Rte = bsxfun(@plus, init_delta*1000 * rand(M, K), 100*h_mu);
    matGamma = matGamma_Shp ./ matGamma_Rte;
    matGamma_Shp(usr_zeros,:) = 0;
    matGamma_Rte(usr_zeros,:) = 0;
    matGamma(usr_zeros,:) = 0;
    
    matDelta_Shp = bsxfun(@plus, init_delta*1000 * rand(N, K), h_pi);
    matDelta_Rte = bsxfun(@plus, init_delta*1000 * rand(N, K), 100*h_pi);
    matDelta = matDelta_Shp ./ matDelta_Rte;
    matDelta_Shp(itm_zeros, :) = 0;
    matDelta_Rte(itm_zeros, :) = 0;
    matDelta(itm_zeros, :) = 0;  
    
    vec_matR_ui = 1/K * sum(matGamma(is_X_train,:) .* matDelta(js_X_train,:), 2);
    vec_matR_ui_shp = g_R + vec_matR_ui .* (log(vec_matR_ui) - psi(vec_matR_ui));
    vec_matR_ui_rte = g_R ./ h_R + init_delta*1000 * rand(length(is_X_train), 1);
    
    vec_matD_ui_shp = vec_matR_ui + vs_X_train;
    vec_matD_ui_rte = vec_matR_ui + sum(matTheta(is_X_train,:) .* matBeta(js_X_train,:), 2);
    vec_matD_ui = ones(length(vec_matR_ui), 1);
    
    vecMu_Shp = g_mu + init_delta * rand(M, 1);
    vecMu_Rte = g_mu / h_mu + init_delta/1e5 * rand(M, 1);
    vecMu = vecMu_Shp ./ vecMu_Rte;
    
    vecPi_Shp = g_pi + init_delta * rand(N, 1);
    vecPi_Rte = g_pi / h_pi + init_delta/1e5 * rand(N, 1);
    vecPi = vecPi_Shp ./ vecPi_Rte;
end