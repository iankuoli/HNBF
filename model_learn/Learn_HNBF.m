 %
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_HNBF(lr)
    
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
            
    global G_prior
    
    a = G_prior(1);
    b = G_prior(2);
    c = G_prior(3);
    d = G_prior(4);
    e = G_prior(5);
    f = G_prior(6);
    g_R_zero = G_prior(7);
    h_R_zero = G_prior(8);
    g_R_nz = G_prior(9);
    h_R_nz = G_prior(10);
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);

    
    %% Estimate weights among the factors
    tmpU = psi(G_matTheta_Shp) - log(G_matTheta_Rte);
    tmpV = psi(G_matBeta_Shp) - log(G_matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update Dispersion
    vec_matD_ui_shp = (1-lr) * vec_matD_ui_shp + lr * (vec_matR_ui + matX_train);
    vec_matD_ui_rte = (1-lr) * vec_matD_ui_rte + lr * (vec_matR_ui + G_matTheta * G_matBeta');
    
    vec_matD_ui = vec_matD_ui_shp ./ vec_matD_ui_rte;
    vec_matD_ui_shp(isinf(vec_matD_ui_shp)) = 0;
    vec_matD_ui_shp(isnan(vec_matD_ui_shp)) = 0;
    vec_matD_ui_rte(isinf(vec_matD_ui_rte)) = 0;
    vec_matD_ui_rte(isnan(vec_matD_ui_rte)) = 0;
    vec_matD_ui(isinf(vec_matD_ui)) = 0;
    vec_matD_ui(isnan(vec_matD_ui)) = 0;
    
    
    %% Update Failure exposure 
    tmp = g_R_zero + vec_matR_ui .* (log(vec_matR_ui) - psi(vec_matR_ui));
    tmp((js_X_train-1)*M + is_X_train) = tmp((js_X_train-1)*M + is_X_train) - g_R_zero + g_R_nz;
    vec_matR_ui_shp = (1-lr) * vec_matR_ui_shp + lr * tmp;
    
    tmp = g_R_zero/h_R_zero + vec_matD_ui - log(vec_matD_ui) - 1;
    tmp((js_X_train-1)*M + is_X_train) = tmp((js_X_train-1)*M + is_X_train) - g_R_zero/h_R_zero + g_R_nz/h_R_nz;
    vec_matR_ui_rte = (1-lr) * vec_matR_ui_rte + lr * tmp;
    
    vec_matR_ui = vec_matR_ui_shp ./ vec_matR_ui_rte;
    vec_matR_ui_shp(isinf(vec_matR_ui_shp)) = 0;
    vec_matR_ui_shp(isnan(vec_matR_ui_shp)) = 0;
    vec_matR_ui_rte(isinf(vec_matR_ui_rte)) = 0;
    vec_matR_ui_rte(isnan(vec_matR_ui_rte)) = 0;
    vec_matR_ui(isinf(vec_matR_ui)) = 0;
    vec_matR_ui(isnan(vec_matR_ui)) = 0;
    
    
    %% Update G_matTheta & G_matBeta
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        G_matTheta_Shp(:, k) = (1-lr) * G_matTheta_Shp(:, k) + lr * (a + sum(tensorPhi, 2));
        G_matBeta_Shp(:, k) = (1-lr) * G_matBeta_Shp(:, k) + lr * (d + sum(tensorPhi, 1)');
    end   
    
    
    %% Updating Latent Factors for Data Modeling --------------------------    
    G_matTheta_Rte = (1-lr) * G_matTheta_Rte + lr * bsxfun(@plus, vec_matD_ui * G_matBeta, G_matEpsilon);
    G_matTheta = G_matTheta_Shp ./ G_matTheta_Rte;
    
    G_matBeta_Rte = (1-lr) * G_matBeta_Rte + lr * bsxfun(@plus, vec_matD_ui' * G_matTheta, G_matEta);
    G_matBeta = G_matBeta_Shp ./ G_matBeta_Rte;
    
    
    %% Update G_vecGamma & G_vecDelta
    G_matEpsilon_Shp = (1-lr) * G_matEpsilon_Shp + lr * (b + K * a);
    G_matEpsilon_Rte = (1-lr) * G_matEpsilon_Rte + lr * (c + sum(G_matTheta, 2));
    G_matEpsilon = G_matEpsilon_Shp ./ G_matEpsilon_Rte;

    G_matEta_Shp = (1-lr) * G_matEta_Shp + lr * (e + K * d);
    G_matEta_Rte = (1-lr) * G_matEta_Rte + lr * (f + sum(G_matBeta, 2));
    G_matEta = G_matEta_Shp ./ G_matEta_Rte;
     
end