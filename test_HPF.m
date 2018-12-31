
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta

% ----- HPF -----
global HPF_matTheta        % dim(M, K): latent document-topic intensities
global HPF_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
global HPF_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

global HPF_matBeta         % dim(N, K): latent word-topic intensities
global HPF_matBeta_Shp     % dim(M, K): varational param of matBeta (shape)
global HPF_matBeta_Rte     % dim(M, K): varational param of matBeta (rate)

global HPF_matEpsilon      % dim(N, K): latent word-topic intensities
global HPF_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
global HPF_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

global HPF_matEta          % dim(N, K): latent word-topic intensities
global HPF_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
global HPF_matEta_Rte      % dim(N, 1): varational param of matEta (rate)

global HPF_prior

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation


global matPrecNRecall



%% Experimental Settings

env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', 'MovieLens100K', 'MovieLens1M', ...
           'LastFm2K', 'LastFm1K', 'EchoNest', 'LastFm360K_2K', 'LastFm360K', ...
           'ML100KPos'};
ENV = env_type{1};
DATA = dataset{5};

NUM_RUNS = 10;
likelihood_step = 10;
check_step = 50;

switch DATA
    case 'SmallToy'
        [ M, N ] = LoadUtilities('/home/ian/Dataset/SmallToy_train.csv', '/home/ian/Dataset/SmallToy_test.csv', '/home/ian/Dataset/SmallToy_valid.csv');
    case 'SmallToyML'
        Ks = [4];
        topK = [5, 10, 15, 20];
    case 'MovieLens100K'
        %Ks = [5, 10, 15, 20];
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 200;
    case 'MovieLens1M'
        %Ks = [5, 10, 15, 20];
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 200;
    case 'LastFm2K'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [100];
        topK = [5, 10, 15, 20];
        MaxItr = 200;
    case 'LastFm1K'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [5,20,50,70,100];
        topK = [5, 10, 15, 20];
        MaxItr = 200;
    case 'LastFm360K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 100;
    case 'LastFm360K_2K'
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 100;
    case 'ML100KPos'
        Ks = [20];
        topK = [5, 10, 15, 20];
    case 'EchoNest'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20];
        MaxItr = 600;
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
    case 'MovieLens100K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens100K_train_v2.csv'), strcat(env_path, 'MovieLens100K_test_v2.csv'), strcat(env_path, 'MovieLens100K_valid_v2.csv'));
    case 'MovieLens1M'
        [ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens1M_train.csv'), strcat(env_path, 'MovieLens1M_test.csv'), strcat(env_path, 'MovieLens1M_valid.csv'));
    case 'LastFm2K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm2K_train.csv'), strcat(env_path, 'LastFm2K_test.csv'), strcat(env_path, 'LastFm2K_valid.csv'));
        %[is_X_train, js_X_train, vs_X_train] = find(matX_train);
        %matX_train = sparse(is_X_train, js_X_train, ones(length(is_X_train),1), M, N);
    case 'LastFm1K'
        [ M, N ] = LoadUtilities(strcat(env_path, 'LastFm1K_train.csv'), strcat(env_path, 'LastFm1K_test.csv'), strcat(env_path, 'LastFm1K_valid.csv'));
        %[is_X_train, js_X_train, vs_X_train] = find(matX_train);
        %matX_train = sparse(is_X_train, js_X_train, ones(length(is_X_train),1), M, N);
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


% Binary version
%[is_X_train, js_X_train] = find(matX_train);
%matX_train = sparse(is_X_train, js_X_train, ones(length(is_X_train),1), M, N);


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

        loglikelihood = zeros(ceil(MaxItr/likelihood_step), 1);


        %% Model initialization
        %               
        newHPF(ini_scale, usr_zeros, itm_zeros, prior);
        
        [is_X_train, js_X_train, vs_X_train] = find(matX_train);
        
        itr = 0;
        IsConverge = false;
        while IsConverge == false
            itr = itr + 1;
            fprintf('Itr: %d  K = %d  ---------------------------------- \n', itr, K);

            % Sample usr_idx, itm_idx
            usr_idx = 1:M;
            itm_idx = 1:N;
            usr_idx(sum(matX_train(usr_idx,:),2)==0) = [];
            itm_idx(sum(matX_train(:,itm_idx),1)==0) = [];
            usr_idx_len = length(usr_idx);
            itm_idx_len = length(itm_idx);

            %% Train HPF
            HPF_Learn();
            

            %% Validation & Testing
            if likelihood_step > 0 && mod(itr, likelihood_step) == 0
                tmpX = sum(HPF_matTheta(is_X_train,:) .* HPF_matBeta(js_X_train,:), 2);
                obsrv_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX);
                loglikelihood(itr / likelihood_step, 1) = mean(obsrv_poisson);
                
                fprintf('Log likelihood of Poisson: %f\n', mean(obsrv_poisson));
            end
            if check_step > 0 && mod(itr, check_step) == 0
                
                fprintf('Validation ... \n');
                indx = itr / check_step;
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
                else
                    user_probe = usr_idx;
                end
                if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Evaluate_ALL(matX_valid(user_probe,:), matX_train(user_probe,:), HPF_matTheta(user_probe,:), HPF_matBeta, topK);
                    fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
                else
                    [valid_precision(indx,:), valid_recall(indx,:), valid_MRR(indx,:)] = Evaluate_PrecNRec(matX_valid(user_probe,:), matX_train(user_probe,:), HPF_matTheta(user_probe,:), HPF_matBeta, topK);
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
                    [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Evaluate_ALL(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), HPF_matTheta(user_probe,:), HPF_matBeta, topK);
                    fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
                else
                    [test_precision(indx,:), test_recall(indx,:), test_MRR(indx,:)] = Evaluate_PrecNRec(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), HPF_matTheta(user_probe,:), HPF_matBeta, topK);
                end
                fprintf('testing precision: %f\n', test_precision(indx,1));
                fprintf('testing recall: %f\n', test_recall(indx,1));
                
                
                range = min(N, 100);
                index = 30;
                tmp1 = HPF_matBeta(1:range,:)*HPF_matTheta(index,:)';
                plot(full([tmp1 matX_train(index,1:range)' matX_test(index,1:range)']));
            end

            if itr >= MaxItr
                IsConverge = true;
            end
        end

        Best_matTheta = HPF_matTheta;
        Best_matBeta = HPF_matBeta;
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
            [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Evaluate_ALL(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), Best_matTheta(usr_idx,:), Best_matBeta, topK);
            fprintf('testing nDCG: %f\n', total_test_nDCG);
        else
            [total_test_precision, total_test_recall, total_test_MRR] = Evaluate_PrecNRec(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), Best_matTheta(usr_idx,:), Best_matBeta, topK);
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











