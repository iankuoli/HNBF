%
% Learning phase of Generator
% The Parameters of generator are updated by (stochastic) variaitonal inference
% J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
% size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
%
function [] = Learn_ExpoMF(usr_idx, itm_idx)
        
    global matA_vs         % expected exposures for all sampled entries
    global matTheta        % dim(M, K): user factors
    global matBeta         % dim(N, K): item factors
    global vecMu           % dim(N, 1): exposure priors
    global valLambda_y     % variance of data Y
    global valLambda_theta % variance of matrix matTheta
    global valLambda_beta  % variance of matrix matbeta
    global val_alpha1
    global val_alpha2
    global K                 % Dimensionality of latent factors
    
    global matX_train
    global matX_train_binary
    
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
    
    % E-step
    vec_prob = sqrt(valLambda_y / (2*pi)) * exp( -valLambda_y * matTheta(usr_idx,:) * matBeta(itm_idx,:)' .^ 2 / 2);
    matA_vs = bsxfun(@times, vec_prob, vecMu(itm_idx)');
    matA_vs = matA_vs ./ bsxfun(@plus, matA_vs, (1-vecMu(itm_idx)'));       % Eq. (4) in the paper
    matA_vs = matA_vs - matA_vs .* matX_train_binary(usr_idx, itm_idx) + matX_train_binary(usr_idx, itm_idx);   % The authors define p_{ui} = 1 if y_{ui} = 1
    
    % M-step
    tmp_py = matA_vs .* matX_train(usr_idx, itm_idx);
    
    tmp_theta_mean = valLambda_y * tmp_py * matBeta(itm_idx,:);
    for u = 1:usr_idx_len
        inv_matrix = valLambda_y * matBeta(itm_idx,:)'* spdiags(matA_vs(u,:)', 0, itm_idx_len, itm_idx_len) * matBeta(itm_idx,:) + ...
                     valLambda_theta * speye(K);
        matTheta(usr_idx(u),:) = tmp_theta_mean(u,:) / inv_matrix;
    end
    tmp_beta_mean = valLambda_y * tmp_py' * matTheta(usr_idx,:);
    for i = 1:itm_idx_len
        inv_matrix = valLambda_y * matTheta(usr_idx,:)'* spdiags(matA_vs(:,i), 0, usr_idx_len, usr_idx_len) * matTheta(usr_idx,:) + ...
                     valLambda_beta * speye(K);
        matBeta(itm_idx(i),:) = tmp_beta_mean(i,:) / inv_matrix;
    end
    
    % Update priors \mu_i
    vecMu(itm_idx) = (val_alpha1 + sum(matA_vs, 1)' - 1) / (val_alpha1 + val_alpha2 + usr_idx_len - 2);
end