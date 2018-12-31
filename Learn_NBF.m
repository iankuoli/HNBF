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
    
    a = G_prior(1);
    b = G_prior(2);
    c = G_prior(3);
    d = G_prior(4);
    e = G_prior(5);
    f = G_prior(6);
    alpha_D = G_prior(7);
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);

    
    %% Estimate weights among the factors
    tmpU = psi(G_matTheta_Shp) - log(G_matTheta_Rte);
    tmpV = psi(G_matBeta_Shp) - log(G_matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update G_matTheta & G_matBeta and G _matThetaX & G_matBetaX   
    matD = (matX_train + alpha_D) ./ (G_matTheta * G_matBeta' + alpha_D);
    
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        G_matTheta_Shp(:, k) = a + sum(tensorPhi, 2);
        G_matBeta_Shp(:, k) = d + sum(tensorPhi, 1)';
    end   
        
    
    %% Updating Latent Factors for Data Modeling --------------------------

    G_matTheta_Rte = bsxfun(@plus, matD * G_matBeta, G_matEpsilon);
    G_matTheta = G_matTheta_Shp ./ G_matTheta_Rte;
    
    G_matBeta_Rte = bsxfun(@plus, matD' * G_matTheta, G_matEta);
    G_matBeta = G_matBeta_Shp ./ G_matBeta_Rte;
    
    
    %% Update G_vecGamma & G_vecDelta
    G_matEpsilon_Shp = b + K * a;
    G_matEpsilon_Rte = c + sum(G_matTheta, 2);
    G_matEpsilon = G_matEpsilon_Shp ./ G_matEpsilon_Rte;

    G_matEta_Shp = e + K * d;
    G_matEta_Rte = f + sum(G_matBeta, 2);
    G_matEta = G_matEta_Shp ./ G_matEta_Rte;
    
    
end