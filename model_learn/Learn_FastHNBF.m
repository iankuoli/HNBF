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
    
    global vec_matR_ui_shp
    global vec_matR_ui_rte
    global vec_matR_ui
    global vec_matD_ui_shp
    global vec_matD_ui_rte
    global vec_matD_ui
    
    global vecMu           % dim(M, 1): approximate matD
    global vecMu_Shp       % dim(M, 1): approximate matD
    global vecMu_Rte       % dim(M, 1): approximate matD
    global matGamma        % dim(M, K): approximate matD
    global matGamma_Shp    % dim(M, K): approximate matD
    global matGamma_Rte    % dim(M, K): approximate matD
    
    global vecPi           % dim(N, 1): approximate matD
    global vecPi_Shp       % dim(N, 1): approximate matD
    global vecPi_Rte       % dim(N, 1): approximate matD
    global matDelta        % dim(N, K): approximate matD
    global matDelta_Shp    % dim(N, K): approximate matD
    global matDelta_Rte    % dim(N, K): approximate matD
            
    global prior
    
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
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);

    
    %% Estimate weights among the factors
    tmpU = psi(matTheta_Shp) - log(matTheta_Rte);
    tmpV = psi(matBeta_Shp) - log(matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update matTheta & matBeta and G _matThetaX & matBetaX   
    tmp_inference = sum(matTheta(is_X_train,:) .* matBeta(js_X_train,:), 2);
    
    h_R = min(mean(vs_X_train) ^ 2 / var(tmp_inference), mean(vs_X_train)/3);
    g_R = h_R;
    
    vec_matD_ui_shp = (1-lr) * vec_matD_ui_shp + lr * (vec_matR_ui + vs_X_train);
    vec_matD_ui_rte = (1-lr) * vec_matD_ui_rte + lr * (vec_matR_ui + tmp_inference);
    vec_matD_ui = vec_matD_ui_shp ./ vec_matD_ui_rte;
    
    vec_matR_ui_shp = (1-lr) * vec_matR_ui_shp + lr * (g_R + vec_matR_ui .* (log(vec_matR_ui) - psi(vec_matR_ui)));
    vec_matR_ui_rte = (1-lr) * vec_matR_ui_rte + lr * (g_R ./ h_R + vec_matD_ui - log(vec_matD_ui) - 1);
    vec_matR_ui = vec_matR_ui_shp ./ vec_matR_ui_rte;
    
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        matTheta_Shp(:, k) = (1-lr) * matTheta_Shp(:, k) + lr * (a + sum(tensorPhi, 2));
        matBeta_Shp(:, k) = (1-lr) * matBeta_Shp(:, k) + lr * (d + sum(tensorPhi, 1)');
        
        matGamma_Shp(:, k) = (1-lr) * matGamma_Shp(:, k) + lr * vecMu;
        matDelta_Shp(:, k) = (1-lr) * matDelta_Shp(:, k) + lr * vecPi;
    end   
    
    
    %% Updating Latent Factors for Data Modeling --------------------------
    
    tmpD = sparse(is_X_train, js_X_train, vec_matD_ui - 1/K*sum(matGamma(is_X_train,:) .* matDelta(js_X_train,:),2), M, N);    
    matTheta_Rte = (1-lr) * matTheta_Rte + lr * bsxfun(@plus, 1/K * (matGamma * (matDelta' * matBeta)) + tmpD * matBeta, matEpsilon);
    matTheta = matTheta_Shp ./ matTheta_Rte;
    
    tmpD = sparse(is_X_train, js_X_train, vec_matD_ui - 1/K*sum(matGamma(is_X_train,:) .* matDelta(js_X_train,:),2), M, N);
    matBeta_Rte = (1-lr) * matBeta_Rte + lr * bsxfun(@plus, 1/K * (matDelta * (matGamma' * matTheta)) + tmpD' * matTheta, matEta);
    matBeta = matBeta_Shp ./ matBeta_Rte;
    
    
    %% Update vecGamma & vecDelta
    matEpsilon_Shp = (1-lr) * matEpsilon_Shp + lr * (b + K * a);
    matEpsilon_Rte = (1-lr) * matEpsilon_Rte + lr * (c + sum(matTheta, 2));
    matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

    matEta_Shp = (1-lr) * matEta_Shp + lr * (e + K * d);
    matEta_Rte = (1-lr) * matEta_Rte + lr * (f + sum(matBeta, 2));
    matEta = matEta_Shp ./ matEta_Rte;
    
    
    %% Updating Latent Factors for Dispersion -----------------------------
    %tmp_inference = sum(matTheta(is_X_train,:) .* matBeta(js_X_train,:), 2);
    tmpD = sparse(is_X_train, js_X_train, tmp_inference, M, N);
    
    matGamma_Rte = (1-lr) * matGamma_Rte + lr * max(bsxfun(@plus, 1/K * (matTheta * (matBeta' * matDelta) - tmpD * matDelta), vecMu), 1e-20);
    matGamma = matGamma_Shp ./ matGamma_Rte;
    
    vecMu_Shp = (1-lr) * vecMu_Shp + lr * (K*g_mu + K * vecMu .* (log(vecMu) - psi(vecMu)));
    vecMu_Rte = (1-lr) * vecMu_Rte + lr * max(K*g_mu ./ h_mu + sum(matGamma - log(matGamma), 2) - K, 1e-20);
    vecMu = vecMu_Shp ./ vecMu_Rte;
    
    matDelta_Rte = (1-lr) * matDelta_Rte + lr * max(bsxfun(@plus, 1/K * (matBeta * (matTheta' * matGamma) - tmpD' * matGamma), vecPi), 1e-20);
    matDelta = matDelta_Shp ./ matDelta_Rte;
    
    vecPi_Shp = (1-lr) * vecPi_Shp + lr * (K*g_pi + K * vecPi .* (log(vecPi) - psi(vecPi)));
    vecPi_Rte = (1-lr) * vecPi_Rte + lr * max(K*g_pi ./ h_pi + sum(matDelta - log(matDelta), 2) - K, 1e-20);
    vecPi = vecPi_Shp ./ vecPi_Rte;
     
end