 %
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_FastHNBF(lr)
    
    global K                 % number of topics
    global M
    global N

    global matX_train        % dim(M, N): consuming records for training
    
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
    
    global vec_matR_ui_shp
    global vec_matR_ui_rte
    global vec_matR_ui
    global vec_matD_ui_shp
    global vec_matD_ui_rte
    global vec_matD_ui
    
    global G_vecMu           % dim(M, 1): approximate matD
    global G_vecMu_Shp       % dim(M, 1): approximate matD
    global G_vecMu_Rte       % dim(M, 1): approximate matD
    global G_matGamma        % dim(M, K): approximate matD
    global G_matGamma_Shp    % dim(M, K): approximate matD
    global G_matGamma_Rte    % dim(M, K): approximate matD
    
    global G_vecPi           % dim(N, 1): approximate matD
    global G_vecPi_Shp       % dim(N, 1): approximate matD
    global G_vecPi_Rte       % dim(N, 1): approximate matD
    global G_matDelta        % dim(N, K): approximate matD
    global G_matDelta_Shp    % dim(N, K): approximate matD
    global G_matDelta_Rte    % dim(N, K): approximate matD
            
    global G_prior
    
    a = G_prior(1);
    b = G_prior(2);
    c = G_prior(3);
    d = G_prior(4);
    e = G_prior(5);
    f = G_prior(6);
    g_mu = G_prior(7);
    h_mu = G_prior(8);
    g_pi = G_prior(9);
    h_pi = G_prior(10);
    g_R = G_prior(11);
    h_R = G_prior(12);
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);

    
    %% Estimate weights among the factors
    tmpU = psi(G_matTheta_Shp) - log(G_matTheta_Rte);
    tmpV = psi(G_matBeta_Shp) - log(G_matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update G_matTheta & G_matBeta and G _matThetaX & G_matBetaX   
    tmp_inference = sum(G_matTheta(is_X_train,:) .* G_matBeta(js_X_train,:), 2);
    
    vec_matD_ui_shp = (1-lr) * vec_matD_ui_shp + lr * (vec_matR_ui + vs_X_train);
    vec_matD_ui_rte = (1-lr) * vec_matD_ui_rte + lr * (vec_matR_ui + tmp_inference);
    vec_matD_ui = vec_matD_ui_shp ./ vec_matD_ui_rte;
    
    vec_matR_ui_shp = (1-lr) * vec_matR_ui_shp + lr * (g_R + vec_matR_ui .* (log(vec_matR_ui) - psi(vec_matR_ui)));
    vec_matR_ui_rte = (1-lr) * vec_matR_ui_rte + lr * (g_R ./ h_R + vec_matD_ui - log(vec_matD_ui) - 1);
    vec_matR_ui = vec_matR_ui_shp ./ vec_matR_ui_rte;
    
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        G_matTheta_Shp(:, k) = (1-lr) * G_matTheta_Shp(:, k) + lr * (a + sum(tensorPhi, 2));
        G_matBeta_Shp(:, k) = (1-lr) * G_matBeta_Shp(:, k) + lr * (d + sum(tensorPhi, 1)');
        
        G_matGamma_Shp(:, k) = (1-lr) * G_matGamma_Shp(:, k) + lr * G_vecMu;
        G_matDelta_Shp(:, k) = (1-lr) * G_matDelta_Shp(:, k) + lr * G_vecPi;
    end   
    
    
    %% Updating Latent Factors for Data Modeling --------------------------
    
    tmpD = sparse(is_X_train, js_X_train, vec_matD_ui - 1/K*sum(G_matGamma(is_X_train,:) .* G_matDelta(js_X_train,:),2), M, N);    
    G_matTheta_Rte = (1-lr) * G_matTheta_Rte + lr * bsxfun(@plus, 1/K * (G_matGamma * (G_matDelta' * G_matBeta)) + tmpD * G_matBeta, G_matEpsilon);
    G_matTheta = G_matTheta_Shp ./ G_matTheta_Rte;
    
    tmpD = sparse(is_X_train, js_X_train, vec_matD_ui - 1/K*sum(G_matGamma(is_X_train,:) .* G_matDelta(js_X_train,:),2), M, N);
    G_matBeta_Rte = (1-lr) * G_matBeta_Rte + lr * bsxfun(@plus, 1/K * (G_matDelta * (G_matGamma' * G_matTheta)) + tmpD' * G_matTheta, G_matEta);
    G_matBeta = G_matBeta_Shp ./ G_matBeta_Rte;
    
    
    %% Update G_vecGamma & G_vecDelta
    G_matEpsilon_Shp = (1-lr) * G_matEpsilon_Shp + lr * (b + K * a);
    G_matEpsilon_Rte = (1-lr) * G_matEpsilon_Rte + lr * (c + sum(G_matTheta, 2));
    G_matEpsilon = G_matEpsilon_Shp ./ G_matEpsilon_Rte;

    G_matEta_Shp = (1-lr) * G_matEta_Shp + lr * (e + K * d);
    G_matEta_Rte = (1-lr) * G_matEta_Rte + lr * (f + sum(G_matBeta, 2));
    G_matEta = G_matEta_Shp ./ G_matEta_Rte;
    
    
    %% Updating Latent Factors for Dispersion -----------------------------
    tmpD = sparse(is_X_train, js_X_train, tmp_inference, M, N);
    
    G_matGamma_Rte = (1-lr) * G_matGamma_Rte + lr * bsxfun(@plus, 1/K * (G_matTheta * (G_matBeta' * G_matDelta) - tmpD * G_matDelta), G_vecMu);
    G_matGamma = G_matGamma_Shp ./ G_matGamma_Rte;
    
    G_vecMu_Shp = (1-lr) * G_vecMu_Shp + lr * (K*g_mu + K * G_vecMu .* (log(G_vecMu) - psi(G_vecMu)));
    G_vecMu_Rte = (1-lr) * G_vecMu_Rte + lr * (K*g_mu ./ h_mu + sum(G_matGamma - log(G_matGamma), 2) - K);
    G_vecMu = G_vecMu_Shp ./ G_vecMu_Rte;
    
    G_matDelta_Rte = (1-lr) * G_matDelta_Rte + lr * bsxfun(@plus, 1/K * (G_matBeta * (G_matTheta' * G_matGamma) - tmpD' * G_matGamma), G_vecPi);
    G_matDelta = G_matDelta_Shp ./ G_matDelta_Rte;
    
    G_vecPi_Shp = (1-lr) * G_vecPi_Shp + lr * (K*g_pi + K * G_vecPi .* (log(G_vecPi) - psi(G_vecPi)));
    G_vecPi_Rte = (1-lr) * G_vecPi_Rte + lr * (K*g_pi ./ h_pi + sum(G_matDelta - log(G_matDelta), 2) - K);
    G_vecPi = G_vecPi_Shp ./ G_vecPi_Rte;
     
end