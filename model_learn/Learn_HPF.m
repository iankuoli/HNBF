 %
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_HPF()
    
    global K                 % number of topics
    global M
    global N

    global matX_train        % dim(M, N): consuming records for training
    
    global HPF_matTheta        % dim(M, K): latent document-topic intensities
    global HPF_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
    global HPF_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

    global HPF_matBeta         % dim(N, K): latent word-topic intensities
    global HPF_matBeta_Shp     % dim(N, K): varational param of matBeta (shape)
    global HPF_matBeta_Rte     % dim(N, K): varational param of matBeta (rate)
    
    global HPF_matEpsilon      % dim(M, 1): latent word-topic intensities
    global HPF_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
    global HPF_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

    global HPF_matEta          % dim(N, 1): latent word-topic intensities
    global HPF_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
    global HPF_matEta_Rte      % dim(N, 1): varational param of matEta (rate)
            
    global HPF_prior
    
    % prior for HPF_matTheta
    a = HPF_prior(1);
    b = HPF_prior(2);
    c = HPF_prior(3);
    
    % prior for HPF_matBeta
    d = HPF_prior(4);
    e = HPF_prior(5);
    f = HPF_prior(6);
        
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    
    
    %% Estimate weights among the factors
    tmpU = psi(HPF_matTheta_Shp) - log(HPF_matTheta_Rte);
    tmpV = psi(HPF_matBeta_Shp) - log(HPF_matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    
    %% Update HPF_matTheta & HPF_matBeta 
    
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        %tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k), M, N);
        HPF_matTheta_Shp(:, k) = a + sum(tensorPhi, 2);
        HPF_matBeta_Shp(:, k) = d + sum(tensorPhi, 1)';
    end
    HPF_matTheta_Rte = bsxfun(@plus, sum(HPF_matBeta), HPF_matEpsilon);
    HPF_matTheta = HPF_matTheta_Shp ./ HPF_matTheta_Rte;
    HPF_matBeta_Rte = bsxfun(@plus, sum(HPF_matTheta), HPF_matEta);
    HPF_matBeta = HPF_matBeta_Shp ./ HPF_matBeta_Rte;
    
    
    %% Update HPF_vecGamma & HPF_vecDelta
    HPF_matEpsilon_Shp = b + K * a;
    HPF_matEpsilon_Rte = c + sum(HPF_matTheta, 2);
    HPF_matEpsilon = HPF_matEpsilon_Shp ./ HPF_matEpsilon_Rte;

    HPF_matEta_Shp = e + K * d;
    HPF_matEta_Rte = f + sum(HPF_matBeta, 2);
    HPF_matEta = HPF_matEta_Shp ./ HPF_matEta_Rte;
    

end