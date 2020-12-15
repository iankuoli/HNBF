
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

% ----- BNMF -----
global BNMF_matTheta
global BNMF_matBeta
global vec_sigmam
% global vec_nlp
global x

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation


global matPrecNRecall



%% Experimental Settings

env_type = {'OSX', 'Linux'};
data_type = {'SmallToy', 'SmallToyML', 'MovieLens100K', 'MovieLens1M', 'LastFm2K', 'LastFm1K', 'ML100KPos'};
ENV = env_type{1};
DATA = data_type{1};

NUM_RUNS = 10;
check_step = 5;


switch DATA
    case 'SmallToy'
        Ks = [4];
        topK = [1, 2, 3, 5];
        MaxItr = 400;
    case 'SmallToyML'
        Ks = [4];
        topK = [1, 2, 3, 5];
        MaxItr = 50;
    case 'MovieLens100K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
    case 'MovieLens1M'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
    case 'LastFm2K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
    case 'LastFm1K'
        Ks = [5, 10, 15, 20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
    case 'ML100KPos'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 400;
end
check_start = 10;
    
if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
	matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*6);
else
    matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*4);
end



%% Load Data

env_path = '';
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
    case 'MovieLens100K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens100K_train_v2.csv'), strcat(env_path, 'MovieLens100K_test_v2.csv'), strcat(env_path, 'MovieLens100K_valid_v2.csv'));
    case 'MovieLens1M'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens1M_train.csv'), strcat(env_path, 'MovieLens1M_test.csv'), strcat(env_path, 'MovieLens1M_valid.csv'));
    case 'LastFm2K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm2K_train.csv'), strcat(env_path, 'LastFm2K_test.csv'), strcat(env_path, 'LastFm2K_valid.csv'));
    case 'LastFm1K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm1K_train.csv'), strcat(env_path, 'LastFm1K_test.csv'), strcat(env_path, 'LastFm1K_valid.csv'));
    case 'ML100KPos'
        [ M, N ] = LoadUtilities(strcat(env_path, 'ml-100k/movielens-100k-train_original.txt'), strcat(env_path, 'ml-100k/movielens-100k-test_original.txt'), strcat(env_path, 'ml-100k/movielens-100k-test_original.txt'));
%         matX_train(matX_train < 4) = 0;
%         matX_train(matX_train > 3.99) = 5;
%         matX_test(matX_test < 4) = 0;
%         matX_test(matX_test > 3.99) = 5;
%         matX_valid(matX_valid < 4) = 0;
%         matX_valid(matX_valid > 3.99) = 5;
end

if max(max(matX_train)) == 10
    matX_train = matX_train / 2;
    matX_test = matX_test / 2;
    matX_valid = matX_valid / 2;
end

usr_zeros = sum(matX_train, 2)==0;
itm_zeros = sum(matX_train, 1)==0;

init_delta = 0.1;
val_sigma = 10;
x = sum(matX_train(:).^2) / 2;

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
        test_precision = zeros(ceil(MaxItr/check_step), length(topK));
        test_recall = zeros(ceil(MaxItr/check_step), length(topK));
        test_nDCG = zeros(ceil(MaxItr/check_step), length(topK));


        %% Model initialization
        %               
        newBNMF(init_delta, usr_zeros, itm_zeros, MaxItr, 1);
        
        itr = 0;
        IsConverge = false;
        total_time = 0;
        while IsConverge == false
            itr = itr + 1;
            fprintf('\nItr: %d  K = %d  ---------------------------------- \n', itr, K);

            % Sample usr_idx, itm_idx
            zero_idx_usr = sum(matX_train,2)==0;
            BNMF_matTheta(zero_idx_usr(:,1),:) = 0;
            zero_idx_itm = sum(matX_train,1)==0;
            BNMF_matTheta(zero_idx_itm(:,1),:) = 0;
            [is_X_train, js_X_train, vs_X_train] = find(matX_train);
            usr_idx = 1:M;
            itm_idx = 1:N;
            usr_idx(zero_idx_usr) = [];
            itm_idx(zero_idx_itm) = [];
            usr_idx_len = length(usr_idx);
            itm_idx_len = length(itm_idx);

            fprintf('subPredict_X: ( %d , %d ) , nnz = %d \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)));


            %% Train generator G
            t = cputime;
            
            % Train generator G given samples and their scores evluated by D
            %     gamma     Volume prior
            %     theta     Prior for val_sigma
            %     sig_k     Prior for val_sigma
            val_sigma = BNMF_Learn(usr_idx, itm_idx, itr, val_sigma, 'reg', 'vol', 'gamma', 1e5, 'theta', 0, 'k', 0);
            vec_sigmam(itr) = val_sigma;
            % vec_nlp(itr) = val_nlp;
            
            total_time = total_time + (cputime - t);

            
            %% Validation & Testing
            if check_step > 0 && mod(itr, check_step) == 0
                
                fprintf('Validation ... \n');
                indx = itr / check_step;
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:)] = Evaluate_ALL(matX_valid, matX_train, BNMF_matTheta, BNMF_matBeta, topK);
                else
                    [valid_precision(indx,:), valid_recall(indx,:)] = Evaluate_PrecNRec(matX_valid, matX_train, BNMF_matTheta, BNMF_matBeta, topK);
                end
                fprintf('validation precision: %f\n', valid_precision(indx,1));
                fprintf('validation recall: %f\n', valid_recall(indx,1));
                fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
                
                
                fprintf('Testing ... \n');
                indx = itr / check_step;
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:)] = Evaluate_ALL(matX_test, matX_train, BNMF_matTheta, BNMF_matBeta, topK);
                else
                    [test_precision(indx,:), test_recall(indx,:)] = Evaluate_PrecNRec(matX_test, matX_train, BNMF_matTheta, BNMF_matBeta, topK);
                end
                fprintf('testing precision: %f\n', test_precision(indx,1));
                fprintf('testing recall: %f\n', test_recall(indx,1));
                fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
            end

            if itr >= MaxItr
                IsConverge = true;
            end
        end
        e = total_time / MaxItr;
        fprintf('computational time per iteration: %f\n', e);
        
       
        QQQ = sum(valid_precision(:,1:2), 2);
        [max_val, max_idx] = max(QQQ(check_start+1:end));
        
        if mod(max_idx, 2) == 1
            if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*3) = [test_precision(max_idx+check_start,:) test_recall(max_idx+check_start,:) test_nDCG(max_idx+check_start,:)];
                matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*3+1:end) = [valid_precision(max_idx+check_start,:) valid_recall(max_idx+check_start,:) valid_nDCG(max_idx+check_start,:)];
            else
                matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*2) = [test_precision(max_idx+check_start,:) test_recall(max_idx+check_start,:)];
                matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*2+1:end) = [valid_precision(max_idx+check_start,:) valid_recall(max_idx+check_start,:)];
            end
        else
            if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*3) = [test_precision(max_idx+check_start,:) test_recall(max_idx+check_start,:) test_nDCG(max_idx+check_start,:)];
                matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*3+1:end) = [valid_precision(max_idx+check_start,:) valid_recall(max_idx+check_start,:) valid_nDCG(max_idx+check_start,:)];
            else
                matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*2) = [test_precision(max_idx+check_start,:) test_recall(max_idx+check_start,:)];
                matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*2+1:end) = [valid_precision(max_idx+check_start,:) valid_recall(max_idx+check_start,:)];
            end
        end
    end
end


save matPrecNRecall











