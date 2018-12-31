
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta
 
% ----- HNBF -----
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

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall



%% Experimental Settings
%
% ---------- Statistics of Datasets ---------- 
% 1. MovieLens100K =>  M = 943     , N = 1682   , NNZ = 100K
% 2. MovieLens1M   =>  M = 6040    , N = 3900   , NNZ = 1M
% 3. LastFm2K      =>  M = 1892    , N = 17632  , NNX = 92,834
% 4. LastFm1K      =>  M = 992     , N = 174091 , NNZ = 898K
% 5. LastFm360K_2K =>  M = 2000    , N = 1682   , NNZ = 
% 6. LastFm360K    =>  M = 359349  , N = 292589 , NNZ = 17,559,486
% 7. ML100KPos     =>  M = 943     , N = 1682   , NNZ = 67,331

env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', 'MovieLens100K', 'MovieLens1M', ...
           'LastFm2K', 'LastFm1K', 'EchoNest', 'LastFm360K_2K', 'LastFm360K', ...
           'ML100KPos'};
ENV = env_type{1};
DATA = dataset{6};

NUM_RUNS = 10;
likelihood_step = 10;
check_step = 10;

ini_scale = 0.001;
data_type = 0; % 1: implicit counts ; 2: ratings; 3: binary implicit
switch DATA
    case 'SmallToy'
        data_type = 1;
        MaxItr = 100;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6, 200, 1e4];
    case 'SmallToyML'
        data_type = 2;
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6, 200, 1e4];
    case 'ML50'
        data_type = 2;
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6, 200, 1e4];
    case 'MovieLens100K'
        data_type = 2;
        MaxItr = 400;
        Ks = [20];
        topK = [5, 10, 15, 20];
        % K=20 fixed setting
        %G_prior = [3, 1, 0.1, ...
        %           3, 1, 0.1, ...
        %           1e2, 1e6, 1e2, 1e6, ...
        %           2, 1e4];
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e6, 2, 1e4];
    case 'MovieLens1M'
        data_type = 2;
        MaxItr = 400;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   1e2, 1e6, 2, 1e4];
    case 'LastFm2K'
        data_type = 1;
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
    case 'LastFm1K'
        data_type = 1;
        Ks = [5,20,50,70,100];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
    case 'EchoNest'
        data_type = 1;
        MaxItr = 400;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
    case 'LastFm360K'
        data_type = 1;
        MaxItr = 400;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
    case 'LastFm360K_2K'
        data_type = 1;
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
    case 'ML100KPos'
        data_type = 3;
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [3, 1, 0.1, ...
                   3, 1, 0.1, ...
                   1e2, 1e8, 200, 1e4];
end
    
if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
	matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*8);
else
    matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*6);
end



%% Load Data
if strcmp(ENV, 'Linux')
    env_path = '/home/ian/Dataset/';
else
    env_path = '/Users/iankuoli/Dataset/';
end

switch DATA
    case 'SmallToy'
        [ M, N ] = LoadUtilities(strcat(env_path, 'SmallToy_train.csv'), strcat(env_path, 'SmallToy_test.csv'), strcat(env_path, 'SmallToy_valid.csv'));
    case 'SmallToyML'
        [ M, N ] = LoadUtilities(strcat(env_path, 'SmallToyML_train.csv'), strcat(env_path, 'SmallToyML_test.csv'), strcat(env_path, 'SmallToyML_valid.csv'));
    case 'ML50'
        [ M, N ] = LoadUtilities(strcat(env_path, 'ML50_train.csv'), strcat(env_path, 'ML50_test.csv'), strcat(env_path, 'ML50_test.csv'));
    case 'MovieLens100K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens100K_train_v2.csv'), strcat(env_path, 'MovieLens100K_test_v2.csv'), strcat(env_path, 'MovieLens100K_valid_v2.csv'));
    case 'MovieLens1M'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens1M_train.csv'), strcat(env_path, 'MovieLens1M_test.csv'), strcat(env_path, 'MovieLens1M_valid.csv'));
    case 'LastFm2K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm2K_train.csv'), strcat(env_path, 'LastFm2K_test.csv'), strcat(env_path, 'LastFm2K_valid.csv'));
    case 'LastFm1K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm1K_train.csv'), strcat(env_path, 'LastFm1K_test.csv'), strcat(env_path, 'LastFm1K_valid.csv'));
    case 'EchoNest'
        [ M, N ] = LoadUtilities(strcat(env_path, 'EchoNest_train.csv'), strcat(env_path, 'EchoNest_test.csv'), strcat(env_path, 'EchoNest_valid.csv'));
    case 'LastFm360K_2K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm360K_2K_train.csv'), strcat(env_path, 'LastFm360K_2K_test.csv'), strcat(env_path, 'LastFm360K_2K_valid.csv'));
    case 'LastFm360K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm360K_train.csv'), strcat(env_path, 'LastFm360K_test.csv'), strcat(env_path, 'LastFm360K_valid.csv'));
    case 'ML100KPos'
        [ M, N ] = LoadUtilities(strcat(env_path, 'ml-100k/movielens-100k-train_original.txt'), strcat(env_path, 'ml-100k/movielens-100k-test_original.txt'), strcat(env_path, 'ml-100k/movielens-100k-test_original.txt'));
        matX_train(matX_train < 4) = 0;
        matX_train(matX_train > 3.99) = 5;
        matX_test(matX_test < 4) = 0;
        matX_test(matX_test > 3.99) = 5;
        matX_valid(matX_valid < 4) = 0;
        matX_valid(matX_valid > 3.99) = 5;
end

if max(max(matX_train)) == 10
    matX_train = matX_train / 2;
    matX_test = matX_test / 2;
    matX_valid = matX_valid / 2;
end

usr_zeros = sum(matX_train, 2)==0;
itm_zeros = sum(matX_train, 1)==0;


%% Experiments

for kk = 1:length(Ks)
    for num = 1:NUM_RUNS


        %% Paramter settings
        %
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
        
        train_poisson = zeros(ceil(MaxItr/likelihood_step), 2);
        test_poisson = zeros(ceil(MaxItr/likelihood_step), 2);
        valid_poisson = zeros(ceil(MaxItr/likelihood_step), 2);

        vecD_tmpX = zeros(ceil(MaxItr/likelihood_step), 3);

        
        %% Model initialization
        %
        switch data_type
            case 1
            % implicit counts
            G_prior(1:6) = [30, 1*K, 0.1*sqrt(K), ...
                            30, 1*K, 0.1*sqrt(K)];
            case 2
            % ratings
            %G_prior(1:6) = [0.3, 0.1*K, 0.1*sqrt(K), ...
            %                0.3, 0.1*K, 0.1*sqrt(K)];
            case 3
            % binary implicit
            G_prior(1:6) = [30, 1*K, 0.1*sqrt(K), ...
                            30, 1*K, 0.1*sqrt(K)];
        end
        newHNBF(ini_scale, usr_zeros, itm_zeros);
        
        [is_X_train, js_X_train, vs_X_train] = find(matX_train);
        
        itr = 0;
        IsConverge = false;
        while IsConverge == false
            itr = itr + 1;
            lr = 1.0;

            % Sample usr_idx, itm_idx
            [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);

            fprintf('Itr: %d  K = %d  ==> ', itr, K);
            fprintf('subPredict_X: ( %d , %d ) , nnz = %d , G_lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), lr);


            %% Train generator G
            %
            % Train generator G given samples and their scores evluated by D
            Learn_HNBF(lr);
            

            %% Calculate log likelihood of Poisson and Negative Binomial
            %
            if likelihood_step > 0 && mod(itr, likelihood_step) == 0
                
                tmpX = sum(G_matTheta(is_X_train,:) .* G_matBeta(js_X_train,:), 2);
                tmpXX = vec_matD_ui((js_X_train-1)*M + is_X_train) .* tmpX;
                vstrain_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX);
                vstrain_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpXX);
                vecD_tmpX(itr / likelihood_step, 1) = mean(vec_matD_ui((js_X_train-1)*M + is_X_train));
                vecD_tmpX(itr / likelihood_step, 2) = mean(tmpX);
                
                [is_X_test, js_X_test, vs_X_test] = find(matX_test);
                tmpX = sum(G_matTheta(is_X_test,:) .* G_matBeta(js_X_test,:), 2);
                tmpXX = vec_matD_ui((js_X_test-1)*M + is_X_test) .* tmpX;
                vstest_poisson = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpX);
                vstest_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpXX);
                
                [is_X_valid, js_X_valid, vs_X_valid] = find(matX_valid);
                tmpX = sum(G_matTheta(is_X_valid,:) .* G_matBeta(js_X_valid,:), 2);
                tmpXX = vec_matD_ui((js_X_valid-1)*M + is_X_valid) .* tmpX;
                vsvalid_poisson = Evaluate_LogLikelihood_Poisson(vs_X_valid, tmpX);
                vsvalid_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_valid, tmpXX);
                        
                fprintf('Train Loglikelihood of Poisson: %f\n', mean(vstrain_poisson));
                fprintf('Train Loglikelihood of NegBinomial: %f\n', mean(vstrain_neg_binomoial));
                fprintf('Valid Loglikelihood of Poisson: %f\n', mean(vsvalid_poisson));
                fprintf('Valid Loglikelihood of NegBinomial: %f\n', mean(vsvalid_neg_binomoial));
                fprintf(' Test Loglikelihood of Poisson: %f\n', mean(vstest_poisson));
                fprintf(' Test Loglikelihood of NegBinomial: %f\n', mean(vstest_neg_binomoial));
                
                l_step_indx = itr/likelihood_step;
                train_poisson(l_step_indx, 1) = mean(vstrain_poisson);
                train_poisson(l_step_indx, 2) = mean(vstrain_neg_binomoial);
                test_poisson(l_step_indx, 1) = mean(vstest_poisson);
                test_poisson(l_step_indx, 2) = mean(vstest_neg_binomoial);
                valid_poisson(l_step_indx, 1) = mean(vsvalid_poisson);
                valid_poisson(l_step_indx, 2) = mean(vsvalid_neg_binomoial);
                
                if l_step_indx > 9 
                    if valid_poisson(l_step_indx,1) > valid_poisson(l_step_indx-1,1)
                        Best_matTheta = G_matTheta;
                        Best_matBeta = G_matBeta;
                    else
                        %IsConverge = true;
                    end
                end
            end
            
            
            %% Calculate precision, recall, MRR, and nDCG
            %
            if check_step > 0 && mod(itr, check_step) == 0
                
                % Calculate the metrics on validation set
                fprintf('Validation ... \n');
                indx = itr / check_step;
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
                else
                    user_probe = usr_idx;
                end
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Evaluate_ALL(matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                                          G_matTheta(user_probe,:), G_matBeta, topK);
                    fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
                else
                    [valid_precision(indx,:), valid_recall(indx,:), valid_MRR(indx,:)] = Evaluate_PrecNRec(matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                           G_matTheta(user_probe,:), G_matBeta, topK);
                end
                fprintf('validation precision: %f\n', valid_precision(indx,1));
                fprintf('validation recall: %f\n', valid_recall(indx,1));
                
                % Calculate the metrics on testing set
                fprintf('Testing ... \n');
                indx = itr / check_step;
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
                else
                    user_probe = 1:length(usr_idx);
                end
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Evaluate_ALL(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                                      G_matTheta(user_probe,:), G_matBeta, topK);
                    fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
                else
                    [test_precision(indx,:), test_recall(indx,:), test_MRR(indx,:)] = Evaluate_PrecNRec(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                        G_matTheta(user_probe,:), G_matBeta, topK);
                end
                fprintf('testing precision: %f\n', test_precision(indx,1));
                fprintf('testing recall: %f\n', test_recall(indx,1));
                
                % Draw a consumption sample 
                range = min(N, 100);
                index = 30;
                tmp1 = G_matBeta(1:range,:) * G_matTheta(index,:)';
                tmp2 = vec_matD_ui(index,1:range)';
                js_sparse = js_X_train(is_X_train == index);
                D_sparse = vec_matD_ui(is_X_train == index);
                js_sparse_range = js_sparse(js_sparse<=range);
                D_sparse_range = D_sparse(js_sparse<=range);
                tmp2(js_sparse_range) = D_sparse_range;
                plot(full([tmp1 tmp1.*tmp2 matX_train(index,1:range)' matX_test(index,1:range)']));
            end
        
            if itr >= MaxItr
                IsConverge = true;
            end               
        end
        
        if valid_poisson(end,1) > valid_poisson(end-1,1)
            Best_matTheta = G_matTheta;
            Best_matBeta = G_matBeta;
        end
        Best_matTheta = G_matTheta;
        Best_matBeta = G_matBeta;
        
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
            [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Evaluate_ALL(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), ...
                                                                                                      Best_matTheta(usr_idx,:), Best_matBeta, topK);
            fprintf('testing nDCG: %f\n', total_test_nDCG);
        else
            [total_test_precision, total_test_recall, total_test_MRR] = Evaluate_PrecNRec(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), ...
                                                                                          Best_matTheta(usr_idx,:), Best_matBeta, topK);
        end
        
        fprintf('total testing precision: %f\n', total_test_precision);
        fprintf('total testing recall: %f\n', total_test_recall);
          
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
            matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [total_test_precision total_test_recall total_test_MRR total_test_nDCG];
            matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_MRR(end,:) valid_nDCG(end,:)];
        else
            matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*3) = [total_test_precision total_test_recall total_test_MRR];
            matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*3+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_MRR(end,:)];
        end
        
    end
end


save matPrecNRecall











