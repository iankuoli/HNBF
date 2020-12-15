function [] = Learn_WMF(alpha, lambda, usr_idx, itm_idx)
    
    global K                 % number of topics

    global matX_train        % dim(M, N): consuming records for training
    global matX_train_binary % dim(M, N): binary consuming records for training
    global matCP
    
    global G_matTheta        % dim(M, K): latent document-topic intensities
    global G_matBeta         % dim(N, K): latent word-topic intensities

    itm_idx_len = length(itm_idx);
    usr_idx_len = length(usr_idx);
    
    matYY = G_matBeta(itm_idx,:)' * G_matBeta(itm_idx,:);
    matRest = matCP * G_matBeta;
    for u = 1:usr_idx_len
        u_idx = usr_idx(u);
        
        vec_u = alpha * matX_train(u_idx, itm_idx)';
        matC_u = spdiags(vec_u, 0, itm_idx_len, itm_idx_len);   % matrix (C^u - I) in the paper
        
        matInv = matYY + G_matBeta(itm_idx,:)' * matC_u * G_matBeta(itm_idx,:) + lambda * speye(K);
        theta_u = matInv \ matRest(u_idx, :)';
        
        G_matTheta(u_idx, :) = theta_u';
        
        if mod(u, 10000) == 0
            fprintf('User %d finished\n', u);
        end
    end
    
    matXX = G_matTheta(usr_idx,:)' * G_matTheta(usr_idx,:);
    matRest =  matCP' * G_matTheta;
    for i = 1:itm_idx_len
        i_idx = itm_idx(i);
        
        vec_i = alpha * matX_train(usr_idx, i_idx);
        matC_i = spdiags(vec_i, 0, usr_idx_len, usr_idx_len);
        
        matInv = matXX + G_matTheta(usr_idx,:)' * matC_i * G_matTheta(usr_idx,:) + lambda * speye(K);
        beta_u = matInv \ matRest(i_idx, :)';
        
        G_matBeta(i_idx, :) = beta_u';
        
        if mod(i, 10000) == 0
            fprintf('Item %d finished\n', i);
        end
    end
end