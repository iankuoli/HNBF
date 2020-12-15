 %
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_CCPF_HPF(t_i, t_j, tau_i, tau_j, c_param, kappa)
    
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
    a_gamma = G_prior(7);
    a_mu = G_prior(8);
    b_mu = G_prior(9);
    a_delta = G_prior(10);
    a_pi = G_prior(11);
    b_pi = G_prior(12);
    
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    
    
    %% Estimate weights among the factors
    tmpU = psi(G_matTheta_Shp) - log(G_matTheta_Rte);
    tmpV = psi(G_matBeta_Shp) - log(G_matBeta_Rte);
    tmpPhi = exp(tmpU(is_X_train,:) + tmpV(js_X_train,:));
    tmpPhi = bsxfun(@times, tmpPhi, 1./sum(tmpPhi, 2));
    
    tmpXU = psi(G_matGamma_Shp) - log(G_matGamma_Rte);
    tmpXV = psi(G_matDelta_Shp) - log(G_matDelta_Rte);
    tmpPhiX = exp(tmpXU(is_X_train,:) + tmpXV(js_X_train,:));
    tmpPhiX = bsxfun(@times, tmpPhiX, 1./sum(tmpPhiX, 2));
    
    
    %% Update G_matTheta & G_matBeta and G _matThetaX & G_matBetaX   
    for k = 1:K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi(:,k) .* vs_X_train, M, N);
        G_matTheta_Shp(:, k) = (1 - t_i) * G_matTheta_Shp(:, k) +  t_i * (a + sum(tensorPhi, 2));
        G_matBeta_Shp(:, k) = (1 - t_j) * G_matBeta_Shp(:, k) + t_j * (d + sum(tensorPhi, 1)');
    end   
    
    G_matTheta_Rte = (1 - t_i) * G_matTheta_Rte + t_i * bsxfun(@plus, G_matGamma * (G_matDelta' * G_matBeta), G_matEpsilon);
    G_matTheta = G_matTheta_Shp ./ G_matTheta_Rte;
    
    G_matBeta_Rte = (1 - t_j) * G_matBeta_Rte + t_j * bsxfun(@plus, G_matDelta * (G_matGamma' * G_matTheta), G_matEta);
    G_matBeta = G_matBeta_Shp ./ G_matBeta_Rte;
    
    G_matEpsilon_Shp = b + K * a;
    G_matEpsilon_Rte = c + sum(G_matTheta, 2);
    G_matEpsilon = G_matEpsilon_Shp ./ G_matEpsilon_Rte;

    G_matEta_Shp = e + K * d;
    G_matEta_Rte = f + sum(G_matBeta, 2);
    G_matEta = G_matEta_Shp ./ G_matEta_Rte;
    
    
    %% Updating Latent Factors for Coupled Compound PF -----------------------------
    range = [0.1:0.1:10];
    phi_n = 1 - c_param + c_param * exp(-range);
    theta_ij = sum(G_matTheta(is_X_train,:) .* G_matBeta(js_X_train,:), 2);
    lambda_ij = log(sum(G_matGamma(is_X_train,:) .* G_matDelta(js_X_train,:), 2)) * range;
    q_prob = -kappa * psi(theta_ij) * phi_n + vs_X_train * log(kappa * phi_n);
    q_prob = exp(bsxfun(@plus, bsxfun(@plus, q_prob, -gammaln(vs_X_train+1)), ...
                 -gammaln(range+1)) + lambda_ij);
    q_prob(isnan(q_prob)) = 0;
    if sum(sum(isnan(q_prob)))>0
        ddd = find(isnan(sum(q_prob,2)));
    end
    q_prob = q_prob + 1e-30;
    aaaaa = q_prob;
    q_prob = q_prob ./ sum(q_prob,2);
    if sum(sum(isnan(q_prob)))>0
        ddd = find(isnan(sum(q_prob,2)));
    end
    q_prob(isnan(q_prob)) = 0;
    q_prob = q_prob + 1e-30;
    q_prob = q_prob ./ sum(q_prob,2);
    E_n_ij(nonzero_idx) = q_prob * range';
    
    tmpXX = sparse(is_X_train, js_X_train, ones(length(is_X_train),1), M, N);
    
    for k = 1:K        
        tensorPhiX = sparse(is_X_train, js_X_train, tmpPhiX(:,k) .* E_n_ij, M, N);
        G_matGamma_Shp = (1-tau_i) * G_matGamma_Shp + tau_i * (a_gamma + sum(tensorPhiX(sample_user_idx,sample_item_idx), 2);
        G_matDelta_Shp = (1-tau_j) * G_matGamma_Shp + tau_j * (a_delta + sum(tensorPhiX(sample_user_idx,sample_item_idx), 1)';
    end 
    
    matIndx = sparse(matSample_ijv(:,1), matSample_ijv(:,2), ones(size(matSample_ijv,1),1), M, N);
    G_matGamma_Rte(sample_user_idx, :) = (1-tau_i) * G_matGamma_Rte(sample_user_idx, :) + ...
                                         tau_i * bsxfun(@plus, matIndx(sample_user_idx, sample_item_idx) * G_matDelta(sample_item_idx, :), G_vecMu(sample_user_idx));
    G_matGamma(sample_user_idx, :) = G_matGamma_Shp(sample_user_idx, :) ./ G_matGamma_Rte(sample_user_idx, :);
    
    G_matDelta_Rte(sample_item_idx, :) = (1-tau_j) * G_matDelta_Rte(sample_item_idx, :) + ...
                                         tau_j * bsxfun(@plus, matIndx(sample_user_idx, sample_item_idx)' * G_matGamma(sample_user_idx, :), G_vecPi(sample_item_idx));
    G_matDelta(sample_item_idx, :) = G_matDelta_Shp(sample_item_idx, :) ./ G_matDelta_Rte(sample_item_idx, :);
    
    G_vecMu_Shp(sample_user_idx) = (1-tau_i) * G_vecMu_Shp(sample_user_idx) + ...
                                   tau_i * (a_mu + K * a_gamma * ones(length(sample_user_idx),1));
    G_vecMu_Rte(sample_user_idx) = (1-tau_i) * G_vecMu_Rte(sample_user_idx) + ...
                                   tau_i * (b_mu + sum(G_matGamma(sample_user_idx,:), 2));
    G_vecMu(sample_user_idx) = G_vecMu_Shp(sample_user_idx, :) ./ G_vecMu_Rte(sample_user_idx, :);
    
    G_vecPi_Shp(sample_item_idx) = (1-tau_j) * G_vecPi_Shp(sample_item_idx) + ...
                                   tau_j * (a_pi + K * a_delta * ones(length(sample_item_idx),1));
    G_vecPi_Rte(sample_item_idx) = (1-tau_j) * G_vecPi_Rte(sample_item_idx) + ...
                                   tau_j * (b_pi + sum(G_matDelta(sample_item_idx,:), 2));
    G_vecPi(sample_item_idx) = G_vecPi_Shp(sample_item_idx) ./ G_vecPi_Rte(sample_item_idx);
    
    G_prior(7) = a_mu / b_mu * sqrt(mean(E_n_ij)/K);
    G_prior(8) = a_pi / b_pi * sqrt(mean(E_n_ij)/K);
    
    if sum(sum(isnan(G_matGamma)))>0
        aaaa = 1;
    end
end