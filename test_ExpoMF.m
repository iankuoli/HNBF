
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta
 
% ----- ExpoMF -----
global matA_vs       % expected exposures for all sampled entries
global matTheta        % dim(M, K): user factors
global matBeta         % dim(N, K): item factors
global vecMu           % dim(N, 1): exposure priors
global valLambda_y     % variance of data Y
global valLambda_theta % variance of matrix matTheta
global valLambda_beta  % variance of matrix matbeta
global val_alpha1
global val_alpha2

global matX_train        % dim(M, N): consuming records for training
global matX_train_binary
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall

global is_binary


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
dataset = {'SmallToy', 'SmallToyML', 'ML50', ...
           'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm1K', 'LastFm2K', 'LastFm360K', 'LastFm360K_2K', 'EchoNest',  ...
           'ML100KPos', 'Jester2', 'ModCloth', 'EachMovie'};
MODEL = "ExpoMF";
ENV = env_type{1};
DATA = dataset{4};

NUM_RUNS = 5;
likelihood_step = 10;
check_step = 5;

ini_scale = 0.001;
topK = [5, 10, 15, 20, 50];
valLambda_y = 1;
valLambda_theta = 1;
valLambda_beta = 1;
val_alpha1 = 1;
val_alpha2 = 1;
switch DATA
    case 'SmallToy'
        MaxItr = 100;
        Ks = [4];
        topK = [1, 2, 3, 5];
    case 'SmallToyML'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
    case 'ML50'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
    case 'MovieLens100K'
        MaxItr = 30;
        Ks = [20];
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 0;
    case 'MovieLens1M'
        MaxItr = 30;
        Ks = [20];
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 0;
    case 'Jester2'
        MaxItr = 5;
        Ks = [20];
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 0;
    case 'ModCloth'
        MaxItr = 20;
        Ks = [20];
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 0;
    case 'EachMovie'
        MaxItr = 30;
        Ks = [20];
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 0;
    case 'LastFm2K'
        Ks = [20];
        MaxItr = 30;
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e8;
        is_binary = 1;
    case 'LastFm1K'
        Ks = [20];
        MaxItr = 10  ;
        valLambda_y = 0.01;
        valLambda_theta = 0.1;
        valLambda_beta = 0.1;
        val_alpha1 = 1e10;
        is_binary = 1;
    case 'EchoNest'
        MaxItr = 400;
        Ks = [20];
        is_binary = 1;
    case 'LastFm360K'
        MaxItr = 300;
        Ks = [20];
        is_binary = 1;
    case 'LastFm360K_2K'
        MaxItr = 300;
        Ks = [20];
        is_binary = 1;
    case 'ML100KPos'
        MaxItr = 1200;
        Ks = [20];
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

if is_binary == 1
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    matX_train = sparse(is_X_train, js_X_train, ones(length(vs_X_train), 1), M, N);
end


%% Experiments

[is_X_train, js_X_train, vs_X_train] = find(matX_train);
matX_train_binary = sparse(is_X_train, js_X_train, ones(length(is_X_train), 1), M, N);

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
        newExpoMF(ini_scale, 0.1, usr_zeros, itm_zeros);
                
        itr = 0;
        IsConverge = false;
        total_time = 0;
        while IsConverge == false
            itr = itr + 1;
            lr = 1.0;

            % Sample usr_idx, itm_idx
            [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);

            fprintf('Run: %d - Itr: %d  K = %d  ==> ', num, itr, K);
            fprintf('subPredict_X: ( %d , %d ) , nnz = %d , lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), lr);


            %% Train generator G
            %
            % Train generator G given samples and their scores evluated by D
            timer = tic;
            
            Learn_ExpoMF(usr_idx, itm_idx);
            
            total_time = total_time + toc(timer);
            
            
            %% Calculate precision, recall, MRR, and nDCG
            %
            if check_step > 0 && mod(itr, check_step) == 0
                indx = itr / check_step;
                
                % Calculate the metrics on validation set
                [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Validation('validation', DATA, matTheta, matBeta, ...
                                                                                                                    topK, usr_idx, usr_idx_len, itm_idx_len);
                
                % Calculate the metrics on testing set
                [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Validation('probing', DATA, matTheta, matBeta, ...
                                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
            end
        
            if itr >= MaxItr
                IsConverge = true;
            end               
        end
        
        Best_matTheta = matTheta;
        Best_matBeta = matBeta;
        
        [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Validation('testing', DATA, Best_matTheta, Best_matBeta, ...
                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
          
        % Record the experimental result
        matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [total_test_precision total_test_recall total_test_nDCG total_test_MRR];
        matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_nDCG(end,:) valid_MRR(end,:)];
        
        fprintf('%s: MaxItr / Itr = %d / %d\n', MaxItr, itr);
        fprintf('Computing time per epoch : %f sec\n\n', total_time / itr);
    end
end


save matPrecNRecall
