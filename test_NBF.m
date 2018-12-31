
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

% ----- ExpoHPF -----
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

global G_matGamma        % dim(M, 1): latent word-topic intensities
global G_matGamma_Shp    % dim(M, 1): varational param of matEpsilonX (shape)
global G_matGamma_Rte    % dim(M, 1): varational param of matEpsilonX (rate)

global G_matDelta        % dim(N, 1): latent word-topic intensities
global G_matDelta_Shp    % dim(N, 1): varational param of matEtaX (shape)
global G_matDelta_Rte    % dim(N, 1): varational param of matEtaX (rate)

global vec_matR_ui
global vec_matD_ui

global G_prior
global matD

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall



%% Experimental Settings

env_type = {'OSX', 'Linux'};
data_type = {'SmallToy', 'SmallToyML', 'ML50', 'MovieLens100K', 'MovieLens1M', 'LastFm2K', 'LastFm1K', 'LastFm360K_2K', 'LastFm360K', 'ML100KPos'};
ENV = env_type{1};
DATA = data_type{5};

NUM_RUNS = 10;
check_step = 10;

ini_scale = 0.001;
switch DATA
    case 'SmallToy'
        MaxItr = 800;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'SmallToyML'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'ML50'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'MovieLens100K'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'MovieLens1M'
        MaxItr = 400;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e4];
    case 'LastFm2K'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e8];
    case 'LastFm1K'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'LastFm360K'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'LastFm360K_2K'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'ML100KPos'
        MaxItr = 1200;
        Ks = [20];
        topK = [5, 10, 15, 20];
        G_prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
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
    case 'LastFm360K_2K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm360K_2K_train.csv'), strcat(env_path, 'LastFm360K_2K_test.csv'), strcat(env_path, 'LastFm360K_2K_valid.csv'));
    case 'LastFm360K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm360K_train.csv'), strcat(env_path, 'LastFm360K_test.csv'), strcat(env_path, 'LastFm360K_valid.csv'));
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


        %% Model initialization
        %               
        %newExpoHPF2(ini_scale, usr_zeros, itm_zeros);
        newNBF(ini_scale, usr_zeros, itm_zeros);
        
        [is_X_train, js_X_train, vs_X_train] = find(matX_train);
        
        itr = 0;
        IsConverge = false;
        avgGamma = zeros(MaxItr, 1);
        while IsConverge == false
            itr = itr + 1;
            fprintf('\nItr: %d  K = %d  ---------------------------------- \n', itr, K);

            % Sample usr_idx, itm_idx
            [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);

            fprintf('subPredict_X: ( %d , %d ) , nnz = %d , G_lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), lr);


            %% Train generator G
            %
            % Train generator G given samples and their scores evluated by D
            Learn_NBF();
            

            %% Validation & Testing
            %
            if check_step > 0 && mod(itr, check_step) == 0
                
                fprintf('Validation ... \n');
                indx = itr / check_step;
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
                else
                    user_probe = usr_idx;
                end
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Evaluate_ALL(matX_valid(user_probe,:), matX_train(user_probe,:), G_matTheta(user_probe,:), G_matBeta, topK);
                    fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
                else
                    [valid_precision(indx,:), valid_recall(indx,:), valid_MRR(indx,:)] = Evaluate_PrecNRec(matX_valid(user_probe,:), matX_train(user_probe,:), G_matTheta(user_probe,:), G_matBeta, topK);
                end
                fprintf('validation precision: %f\n', valid_precision(indx,1));
                fprintf('validation recall: %f\n', valid_recall(indx,1));
                
                
                fprintf('Testing ... \n');
                indx = itr / check_step;
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
                else
                    user_probe = 1:length(usr_idx);
                end
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Evaluate_ALL(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), G_matTheta(user_probe,:).*G_matGamma(user_probe,:), G_matBeta.*G_matDelta, topK);
                    fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
                else
                    [test_precision(indx,:), test_recall(indx,:), test_MRR(indx,:)] = Evaluate_PrecNRec(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), G_matTheta(user_probe,:), G_matBeta, topK);
                end
                fprintf('testing precision: %f\n', test_precision(indx,1));
                fprintf('testing recall: %f\n', test_recall(indx,1));
                
                %range = min(N, 100);
                %index = 1;
                %tmp1 = G_matBeta(1:range,:)*G_matTheta(index,:)';
                %tmp2 = vec_matD_ui(index, 1:range)';
                %plot(full([tmp1 tmp1.*tmp2 matX_train(index,1:range)' matX_test(index,1:range)']));
            end

            avgGamma(itr,1) = mean(mean(G_matGamma));
            if itr >= MaxItr
                IsConverge = true;
            end
        end
        
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
            matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [test_precision(end,:) test_recall(end,:) test_MRR(end,:) test_nDCG(end,:)];
            matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_MRR(end,:) valid_nDCG(end,:)];
        else
            matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*3) = [test_precision(end,:) test_recall(end,:) test_MRR(end,:)];
            matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*3+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_MRR(end,:)];
        end
    end
end


save matPrecNRecall











