
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta

% ----- PMF -----
global PRPF_matTheta        % dim(M, K): latent document-topic intensities
global PRPF_matTheta_Shp    % dim(M, K): varational param of PRPF_matTheta (shape)
global PRPF_matTheta_Rte    % dim(M, K): varational param of PRPF_matTheta (rate)

global PRPF_matBeta         % dim(N, K): latent word-topic intensities
global PRPF_matBeta_Shp     % dim(M, K): varational param of PRPF_matBeta (shape)
global PRPF_matBeta_Rte     % dim(M, K): varational param of PRPF_matBeta (rate)

global PRPF_matEpsilon      % dim(N, K): latent word-topic intensities
global PRPF_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
global PRPF_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

global PRPF_matEta          % dim(N, K): latent word-topic intensities
global PRPF_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
global PRPF_matEta_Rte      % dim(N, 1): varational param of matEta (rate)

global PRPF_prior

global matX_predict


global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation


global matPrecNRecall



%% Experimental Settings
MODEL = {'pairPRPF', 'list_linearPRPF', 'list_expPRPF', 'pointPRPF'};
env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', ...
           'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm1K', 'LastFm2K', 'LastFm360K', 'LastFm360K_2K', 'EchoNest',  ...
           'ML100KPos', 'Jester2', 'ModCloth', 'EachMovie'};
ENV = env_type{1};
DATA = dataset{9};
LTR = MODEL{1};

NUM_RUNS = 10;
S_BBVI = 5;
R_samples = 6;
topD = 0;    
Ks = [20];
topK = [5, 10, 15, 20, 50];
kappa = 0.5;
delta = 1;
switch DATA
    case 'SmallToy'
        [ M, N ] = LoadUtilities('/home/ian/Dataset/SmallToy_train.csv', '/home/ian/Dataset/SmallToy_test.csv', '/home/ian/Dataset/SmallToy_valid.csv');
    case 'SmallToyML'
        Ks = [4];
        topK = [5, 10, 15, 20];
        kappa = 0.5;
        delta = 1;
        alpha = 100;
        MaxItr = 30;
        check_step = 1;
    case 'MovieLens100K'
        delta = 1;
        alpha = 1000;
        MaxItr = 100;    % for epoch per iteration
        check_step = 1;
        check_start = 6;
        switch LTR
            case 'pointPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'MovieLens1M'
        alpha = 1000;
        MaxItr = 100;
        check_step = 10;
        check_start = 6;
        switch LTR
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'MovieLens20M'
        alpha = 1000;
        MaxItr = 60;
        check_step = 10;
        check_start = 10;
    case 'Jester2'
        delta = 1;
        alpha = 1000;
        MaxItr = 30;    % for epoch per iteration
        check_step = 1;
        check_start = 5;
        switch LTR
            case 'pointPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'ModCloth'
        delta = 1;
        alpha = 1000;
        MaxItr = 10;    % for epoch per iteration
        check_step = 1;
        check_start = 5;
        switch LTR
            case 'pointPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'EachMovie'
        delta = 1;
        alpha = 1000;
        MaxItr = 100;    % for epoch per iteration
        check_step = 1;
        check_start = 5;
        switch LTR
            case 'pointPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'LastFm2K'  
        MaxItr = 100;
        check_step = 5;
        check_start = 1;
        delta = 1;
        alpha = 100;
        switch LTR
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'LastFm1K'
        MaxItr = 5;
        check_step = 5;
        check_start = 1;
        delta = 1;
        alpha = 1000;
        max_consumed_items = 200;
        switch LTR
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        end
    case 'LastFm360K_2K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        kappa = 0.5;
        delta = 1;
        alpha = 100;
        MaxItr = 60;
        check_step = 5;
        check_start = 6;
    case 'LastFm360K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        kappa = 0.5;
        delta = 1;
        alpha = 100;
        MaxItr = 60;
        check_step = 5;
        check_start = 6;
        switch LTR
            case 'pairPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
                alpha = 100;
            case 'list_linearPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
                alpha = 1000;
            case 'list_expPRPF'
                PRPF_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
                alpha = 1000;
        end
    case 'ML100KPos'
        %Ks = [5, 10, 15, 20];
        Ks = [20];
        topK = [5, 10, 15, 20];
        kappa = 0.5;
        delta = 1;
        alpha = 100;
        MaxItr = 60;
        check_step = 5;
        check_start = 6;
end
    
%
% Create Recording Container
% -------------------------------------------------------------------------
matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*8);


%
% Load Data
% -------------------------------------------------------------------------
LoadData(DATA, ENV);
usr_zeros = sum(matX_train, 2)==0;
itm_zeros = sum(matX_train, 1)==0;

if topD > 0
    for u = 1:M
        if sum(matX_train(u, :) > 0) > topD
            [val_u, indx_u] = find(matX_train(u, :));
            [sort_val, sort_idx] = sort(val_u, 'ascend');
            matX_train(u, indx_u(sort_idx(1:(length(sort_idx) - topD)))) = 0;
        end  
    end
end

if strcmp(LTR, 'pointPRPF')
    new_us = [];
    new_is = [];
    new_vals = [];
    for u = 1:M
        [uu, indx_u, val_u] = find(matX_train(u, :));
        [sort_val, sort_idx] = sort(val_u, 'ascend');
        new_val = 1;
        
        for ii = 1:length(sort_val)
            if ii==1 || sort_val(ii) > sort_val(ii-1)
                sort_val(sort_val==sort_val(ii)) = new_val;
                new_val = new_val + 1;
            end
        end
        
        new_us = [new_us, uu*u];
        new_is = [new_is, indx_u(sort_idx)];
        new_vals = [new_vals, sort_val];
    end
    matX_predict = sparse(new_us, new_is, new_vals, M, N);
end


%% Experiments

for kk = 1:length(Ks)
    for num = 1:NUM_RUNS


        %% Paramter settings
        %
        kappa = 0.5;
        Ini = true;
        ini_scale = 0.001;
        
        K = Ks(kk);
        usr_batch_size = M;

        valid_precision = zeros(ceil(MaxItr/check_step), length(topK));
        valid_recall = zeros(ceil(MaxItr/check_step), length(topK));
        valid_nDCG = zeros(ceil(MaxItr/check_step), length(topK));
        valid_MRR = zeros(ceil(MaxItr/check_step), length(topK));
        
        test_precision = zeros(ceil(MaxItr/check_step), length(topK));
        test_recall = zeros(ceil(MaxItr/check_step), length(topK));
        test_nDCG = zeros(ceil(MaxItr/check_step), length(topK));
        test_MRR = zeros(ceil(MaxItr/check_step), length(topK));


        %% Model initialization
        %
                       
        [M, N] = size(matX_train);
        C = mean(sum(matX_train>0,2));
        
        usr_zeros = sum(matX_train, 2)==0;
        itm_zeros = sum(matX_train, 1)==0;
        
        newPRPF(usr_zeros, itm_zeros);
        
        a = PRPF_prior(1);
        b = PRPF_prior(2);
        c = PRPF_prior(3);
        d = PRPF_prior(4);
        e = PRPF_prior(5);
        f = PRPF_prior(6);      
        
        [is_X_train, js_X_train, vs_X_train] = find(matX_train);
        
        itr = 0;
        IsConverge = false;
        total_time = 0;
        while IsConverge == false
            itr = itr + 1;
            fprintf('\nRun: %d , Itr: %d  K = %d  ---------------------------------- \n', num, itr, K);
            t = cputime; 
            %
            % Set the learning rate 
            % ref: Content-based recommendations with Poisson factorization. NIPS, 2014
            %
            if usr_batch_size == M
                SVI_lr = 1;
            else
                offset = 100;
                SVI_lr = 2*(offset + itr) ^ -kappa;
            end

            % Sample usr_idx, itm_idx
            [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);

            fprintf('subPredict_X: ( %d , %d ) , nnz = %d , SVI_lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), SVI_lr);

            %
            % Update matX_predict by the framework of PRPF
            %
            if strcmp(LTR, 'pointPRPF') == false
                predict_X = update_matX_predict_bk2(LTR, usr_idx, itm_idx, delta, alpha, SVI_lr, C);
                plot([matX_train(1,1:100)', predict_X(1, 1:100)' (PRPF_matTheta(1,:)*PRPF_matBeta(1:100,:)')'])
                %[ii, jj, vv] = find(predict_X(1,:));
                %priors = sum(PRPF_matTheta(usr_idx(ii),:) .* PRPF_matBeta(itm_idx(jj),:), 2);
                % plot([vv', priors]);
            end
            
            %% Update Latent Tensor Variables
            %
            % Update tensorPhi
            %
            fprintf('Update (is, js, vs) of tensorPhi ... ');
            [x_i, x_j] = find(matX_predict(usr_idx, itm_idx));

            tmpX = psi(PRPF_matTheta_Shp(usr_idx,:)) - log(PRPF_matTheta_Rte(usr_idx,:));
            tmpY = psi(PRPF_matBeta_Shp(itm_idx,:)) - log(PRPF_matBeta_Rte(itm_idx,:));

            % The code may cause memory error while processing mass data.
            tmpV = exp(tmpX(x_i,:) + tmpY(x_j,:));
            tmpV = bsxfun(@times, tmpV, 1./sum(tmpV, 2));
            
            
            %
            % Update tensorPhi , matTheta_Shp , matTheta_Rte , matBeta_Shp , matBeta_Rte
            %
            if usr_batch_size == M
              scale = ones(length(itm_idx), 1);
            else
              scale = (sum(matX_predict(:, itm_idx) > 0, 1) ./ sum(matX_predict(usr_idx, itm_idx) > 0, 1))';
            end
    
            for k = 1:K
                tensorPhi = sparse(x_i, x_j, tmpV(:,k), usr_idx_len, itm_idx_len);
                PRPF_matTheta_Shp(usr_idx, k) = (1 - SVI_lr) * PRPF_matTheta_Shp(usr_idx, k) + SVI_lr * (a + sum(matX_predict(usr_idx, itm_idx) .* tensorPhi, 2));
                PRPF_matBeta_Shp(itm_idx, k) = (1 - SVI_lr) * PRPF_matBeta_Shp(itm_idx, k) + SVI_lr * (d + scale .* sum(matX_predict(usr_idx, itm_idx) .* tensorPhi, 1)');
            end
            PRPF_matTheta_Rte(usr_idx,:) = (1 - SVI_lr) * PRPF_matTheta_Rte(usr_idx,:) + SVI_lr * bsxfun(@plus, sum(PRPF_matBeta(itm_idx,:), 1), PRPF_matEpsilon(usr_idx));
            PRPF_matTheta(usr_idx,:) = PRPF_matTheta_Shp(usr_idx,:) ./ PRPF_matTheta_Rte(usr_idx,:);

            PRPF_matBeta_Rte(itm_idx, :) = (1 - SVI_lr) * PRPF_matBeta_Rte(itm_idx, :) + SVI_lr * bsxfun(@plus, bsxfun(@times, sum(PRPF_matTheta(usr_idx,:), 1), scale), PRPF_matEta(itm_idx));
            PRPF_matBeta(itm_idx, :) = PRPF_matBeta_Shp(itm_idx, :) ./ PRPF_matBeta_Rte(itm_idx, :);
            
            %
            % Update matEpsilon_Shp , matEpsilon_Rte
            %
            PRPF_matEpsilon_Shp(usr_idx) = (1-SVI_lr) * PRPF_matEpsilon_Shp(usr_idx) + SVI_lr * (b + K * a);
            PRPF_matEpsilon_Rte(usr_idx) = (1-SVI_lr) * PRPF_matEpsilon_Rte(usr_idx) + SVI_lr * (c + sum(PRPF_matTheta(usr_idx,:), 2));
            PRPF_matEpsilon(usr_idx) = PRPF_matEpsilon_Shp(usr_idx) ./ PRPF_matEpsilon_Rte(usr_idx);

            %
            % Update matEta_Shp , matEta_Rte
            %
            PRPF_matEta_Shp(itm_idx) = (1-SVI_lr) * PRPF_matEta_Shp(itm_idx) + SVI_lr * (e + K * d);
            PRPF_matEta_Rte(itm_idx) = (1-SVI_lr) * PRPF_matEta_Rte(itm_idx) + SVI_lr * (f + sum(PRPF_matBeta(itm_idx,:), 2));
            PRPF_matEta(itm_idx) = PRPF_matEta_Shp(itm_idx) ./ PRPF_matEta_Rte(itm_idx);
            
            total_time = total_time + (cputime - t);
            
            %[ii, jj, vv] = find(matX_predict(1,:));
            %priors = sum(PRPF_matTheta(usr_idx(ii),:) .* PRPF_matBeta(itm_idx(jj),:), 2);
            %plot([vv', priors]);figure(gcf);
            
            
            %% Validation & Testing
            if check_step > 0 && mod(itr, check_step) == 0
                indx = itr / check_step;
                
                % Calculate the metrics on validation set
                [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Validation('validation', DATA, PRPF_matTheta, PRPF_matBeta, ...
                                                                                                                    topK, usr_idx, usr_idx_len, itm_idx_len);
                                                                                                                
                % Calculate the metrics on testing set
                [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Validation('probing', DATA, PRPF_matTheta, PRPF_matBeta, ...
                                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
            end
            
            if itr >= MaxItr
                IsConverge = true;
            end
        end
        
        Best_matTheta = PRPF_matTheta;
        Best_matBeta = PRPF_matBeta;
        
        [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Validation('testing', DATA, Best_matTheta, Best_matBeta, ...
                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
          
        % Record the experimental result
        matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [total_test_precision total_test_recall total_test_nDCG total_test_MRR];
        matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_nDCG(end,:) valid_MRR(end,:)];
        
        fprintf('%s: MaxItr / Itr = %d / %d \n', LTR, MaxItr, itr);
        fprintf('Computing time per epoch : %f sec\n\n', total_time / itr);     
    end
end


save matPrecNRecall
