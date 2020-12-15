 %
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_NBF()
    
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
    
    global matD
            
    global prior
    
    a = prior(1);
    b = prior(2);
    c = prior(3);
    d = prior(4);
    e = prior(5);
    f = prior(6);
    alpha_D = prior(7);
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);

    
    %% Estimate weights among the factors
    tmpU = psi(matTheta_Shp) - log(matTheta_Rte);
    tmpV = psi(matBeta_Shp) - log(matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update matTheta & matBeta and G _matThetaX & matBetaX   
    matD = (matX_train + alpha_D) ./ (matTheta * matBeta' + alpha_D);
    
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        matTheta_Shp(:, k) = a + sum(tensorPhi, 2);
        matBeta_Shp(:, k) = d + sum(tensorPhi, 1)';
    end   
        
    
    %% Updating Latent Factors for Data Modeling --------------------------

    matTheta_Rte = bsxfun(@plus, matD * matBeta, matEpsilon);
    matTheta = matTheta_Shp ./ matTheta_Rte;
    
    matBeta_Rte = bsxfun(@plus, matD' * matTheta, matEta);
    matBeta = matBeta_Shp ./ matBeta_Rte;
    
    
    %% Update G_vecGamma & G_vecDelta
    matEpsilon_Shp = b + K * a;
    matEpsilon_Rte = c + sum(matTheta, 2);
    matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

    matEta_Shp = e + K * d;
    matEta_Rte = f + sum(matBeta, 2);
    matEta = matEta_Shp ./ matEta_Rte;
    
    
end