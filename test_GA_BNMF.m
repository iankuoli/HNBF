% 
% Bayesian Probabilistic Matrix Factorization
%
%
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

% ----- Generator -----
global G_matX_tilde

global G_matTheta        % dim(M, K): latent document-topic intensities
global G_matBeta         % dim(N, K): latent word-topic intensities

global G_kappa
global vec_sigmam

% ----- Discriminator -----
global D_matTheta        % dim(M, K): latent document-topic intensities
global D_matBeta         % dim(N, K): latent word-topic intensities

global D_sqGrad_matTheta
global D_sqGrad_matBeta

% ----- Sampling -----
global vecSample_i
global vecSample_j
global matSample_val
global matSample_Score

% ----- Data -----
global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

% ----- Evaluation -----
global matPrecNRecall


global is_useGAN
is_useGAN = false;
is_SN = false;


%% Load Data

env_type = {'OSX', 'Linux'};
data_type = {'SmallToy', 'SmallToyML', 'MovieLens100K', 'MovieLens1M', 'LastFm2K', 'LastFm1K', 'ML100KPos'};
ENV = env_type{1};
DATA = data_type{3};

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
        %[ M, N ] = LoadUtilities(strcat(env_path, 'MovieLens100K_train.csv'), strcat(env_path, 'MovieLens100K_test.csv'), strcat(env_path, 'MovieLens100K_valid.csv'));
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



%% Experimental Settings

%Ks = [5, 10, 15];
Ks = [20];
%Ks = [100];
topK = [5, 10, 15, 20];

NUM_RUNS = 10;
S_BBVI = 6;
R_samples = 6;
check_step = 5;
D_lr_scale = 10;

switch DATA
    case 'SmallToy'
        MaxItr = 400;
        ini_bias = 300;
        G_kappa = 10;
        Ks = [4];
        topK = [5, 10, 15, 20];
        BBVI_lr_scale = 0.001;
        zero_samples_per_user = 10;
        tau = 0.1;
    case 'SmallToyML'
        MaxItr = 200;
        ini_bias = 5;
        G_kappa = 10;
        Ks = [4];
        topK = [5, 10, 15, 20];
        BBVI_lr_scale = 0.01;
        zero_samples_per_user = 10;
        tau = 0.01;
    case 'MovieLens100K'
        MaxItr = 1000;
        ini_bias = 3;
        G_kappa = 2;
        BBVI_lr_scale = 0.05;     % for FisherGAN
        %BBVI_lr_scale = 1;     % for GANforSN
        zero_samples_per_user = 50;
        check_start = 20;
        %tau = 0.1;
        tau = 0.1;
        %tau = nnz(matX_train) / (M * N);
    case 'MovieLens1M'
        MaxItr = 600;
        ini_bias = 5;
        G_kappa = 1;
        BBVI_lr_scale = 0.1;
        zero_samples_per_user = 100;
        check_start = 80;
        tau = 0.01;
    case 'LastFm2K'
        MaxItr = 400;
        ini_bias = 300;
        G_kappa = 1;
        BBVI_lr_scale = 10;
        zero_samples_per_user = 100;
        check_start = 20;
        %tau = 0.01;
        tau = nnz(matX_train) / (M * N);
    case 'LastFm1K'
        MaxItr = 400;
        ini_bias = 400;
        G_kappa = 1;
        BBVI_lr_scale = 10;
        D_lr_scale = 10;
        zero_samples_per_user = 100;
        check_start = 26;    
        %tau = 0.0041;
        tau = nnz(matX_train) / (M * N);
    case 'ML100KPos'
        MaxItr = 300;
        ini_bias = 5;
        G_kappa = 1;
        BBVI_lr_scale = 0.1;
        zero_samples_per_user = 50;
        check_start = 6;
        tau = 0.5;
end
    
if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
	matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*6);
else
    matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*4);
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
        Vlog_likelihood = zeros(ceil(MaxItr/check_step));
        test_precision = zeros(ceil(MaxItr/check_step), length(topK));
        test_recall = zeros(ceil(MaxItr/check_step), length(topK));
        test_nDCG = zeros(ceil(MaxItr/check_step), length(topK));
        Tlog_likelihood = zeros(ceil(MaxItr/check_step));
        
        
        vec_Gtheta_u = rand(M, 1) - 0.5;
        vec_Gtheta_v = rand(K, 1) - 0.5;
        vec_Gbeta_u = rand(N, 1) - 0.5;
        vec_Gbeta_v = rand(K, 1) - 0.5;


        %% Model initialization
        %
        vec_sigmam = zeros(MaxItr, 1);
        if is_SN == true
            newGeneratorBNMF_SN(ini_bias, ini_scale, usr_zeros, itm_zeros);
            newDiscriminator_SN(ini_scale, 0.1, 0.1, usr_zeros, itm_zeros);
        else
            newGeneratorBNMF(ini_bias, ini_scale, usr_zeros, itm_zeros);
            newDiscriminator(ini_scale, 0.1, 0.1, usr_zeros, itm_zeros);
        end
        


        %% Training
        %
        IsConverge = false;
        itr = 0;
        t = cputime;
        val_sigma = 1;
        while IsConverge == false
            itr = itr + 1;
            fprintf('\nItr: %d  K = %d  ---------------------------------- \n', itr, K);

            % Sample usr_idx, itm_idx
            [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData(usr_batch_size);
            [is_X, js_X, vs_X] = find(matX_train(usr_idx, itm_idx));
            nnz_matX_train = nnz(matX_train(usr_idx, itm_idx));

            %
            % Set the learning rate
            % ref: Content-based recommendations with Poisson factorization. NIPS, 2014
            %      Black box variational inference. AISTATS, 2014
            %
            BBVI_lr = BBVI_lr_scale * (1000. + itr) ^ -kappa;
            D_lr = D_lr_scale * (1000. + itr) ^ -kappa;

            fprintf('subPredict_X: ( %d , %d ) , nnz = %d , D_lr = %f , lr(BBVI)_D = %f\n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), D_lr, BBVI_lr);

            % Compute the user-wise normalized weight.
            user_weights = full(sum(sparse(is_X(1:nnz_matX_train), js_X(1:nnz_matX_train), ones(nnz_matX_train,1), usr_idx_len, itm_idx_len), 2));
            user_weights = mean(user_weights) * (1 ./ user_weights);

            % Compute the ratio of non_zeors / neg_samples 
            tauQ = (usr_idx_len * itm_idx_len - nnz_matX_train) / (sample_size - nnz_matX_train);
            %tauQ = mean(user_weights) / (sample_size - nnz_matX_train);
            %tauQ = 1 / (sample_size - nnz_matX_train);
            %tauQ = mean(user_weights);   % for last.fm1K
            %tauQ = 1;   % for MovieLens100K
            %tauQ = 0.1;

            %% Train discriminator D
            %
            if is_useGAN == true
                for itrr = 1:1
                    fprintf('itr_DIS: %d  ', itrr);

                    % Sample data by generator G
                    [vecSample_i, vecSample_j, matSample_val] = G_Sample_BNMF(usr_idx, itm_idx, R_samples, usr_idx_len * zero_samples_per_user, val_sigma);

                    % Use the samples to train discriminator D   
                    if is_SN == true
                        % Based on GANs with spectral normalization and user-wise normalization in IPM.
                        D_Learn_Fisher2_userwise_SpectralNorm(0.1*D_lr, 0.9, usr_idx, itm_idx, user_weights, tauQ);
                    else
                        % Based on Fisher GANs with user-wise normalization in IPM.
                        D_Learn_Fisher2_userwise4(0.1*D_lr, D_lr, 0.9, usr_idx, itm_idx, user_weights, tauQ);
                    end
                end
            end


            %% Train generator G

            % Compute likelihoods (scores) by discriminator D
            if is_useGAN == true
                if is_SN == true
                    [vecSample_i, vecSample_j, matSample_val] = G_Sample_BNMF_SN(usr_idx, itm_idx, R_samples, usr_idx_len * zero_samples_per_user, val_sigma);
                else
                    [vecSample_i, vecSample_j, matSample_val] = G_Sample_BNMF(usr_idx, itm_idx, R_samples, usr_idx_len * zero_samples_per_user, val_sigma);
                end

                sample_size = length(vecSample_i) / R_samples;
                if is_SN == true
                    matSample_Score = D_Evaluate_SN(usr_idx, itm_idx, vecSample_i, vecSample_j, matSample_val);
                else
                    matSample_Score = D_Evaluate4(usr_idx, itm_idx, vecSample_i, vecSample_j, matSample_val);
                end
                fprintf('Samples mean = %f\n', mean(mean(matSample_val)));
            end

            % Train generator G given samples and their scores evluated by D
            if is_SN == true
                val_sigma = G_Learn_BNMF_userwise_SN(BBVI_lr, usr_idx, itm_idx, S_BBVI, R_samples, tau, user_weights, tauQ);
            else
                val_sigma = G_Learn_BNMF_userwise(BBVI_lr, usr_idx, itm_idx, S_BBVI, R_samples, tau, user_weights, tauQ,...
                                                  itr, val_sigma, 'reg', 'dist', 'gamma', 1, 'k', 0, 'theta', 0);
            end
            vec_sigmam(itr) = val_sigma;
            

            %% Validation & Testing
            if check_step > 0 && mod(itr, check_step) == 0
                
                fprintf('Validation ... \n');
                indx = itr / check_step;
                if strcmp(DATA, 'SmallToyML') || strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:)] = Evaluate_ALL(matX_valid, matX_train, G_matTheta, G_matBeta, topK);
                    fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
                else
                    [valid_precision(indx,:), valid_recall(indx,:)] = Evaluate_PrecNRec(matX_valid, matX_train, G_matTheta, G_matBeta, topK);
                end
                fprintf('validation precision: %f\n', valid_precision(indx,1));
                fprintf('validation recall: %f\n', valid_recall(indx,1));                
                
                fprintf('Testing ... \n');
                indx = itr / check_step;
                if strcmp(DATA, 'SmallToyML') || strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                    [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:)] = Evaluate_ALL(matX_test, matX_train, G_matTheta, G_matBeta, topK);
                    fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
                else
                    [test_precision(indx,:), test_recall(indx,:)] = Evaluate_PrecNRec(matX_test, matX_train, G_matTheta, G_matBeta, topK);
                end
                fprintf('testing precision: %f\n', test_precision(indx,1));
                fprintf('testing recall: %f\n', test_recall(indx,1));
            end


            %% Draw    
            if mod(itr, 10) == 0
                vecSample_j_R_samples = repmat(vecSample_j, 1, R_samples);
                for i = 1:R_samples-1
                    vecSample_j_R_samples(:, i+1) = vecSample_j_R_samples(:, i+1) + i * itm_idx_len;
                end
                vecSample_j_R_samples = reshape(vecSample_j_R_samples, [], 1);
                
                matX_sample_score = sparse(repmat(vecSample_i,R_samples,1), vecSample_j_R_samples, reshape(matSample_Score,[],1), usr_idx_len, itm_idx_len * R_samples);
                matX_sample_val = sparse(repmat(vecSample_i,R_samples,1), vecSample_j_R_samples, reshape(matSample_val,[],1), usr_idx_len, itm_idx_len * R_samples);
                %user = 1; item = 5;
                user = 1; item = 18;    % MovieLens100K
                full([matX_sample_val(user,item) matX_sample_score(user,item); matX_sample_val(user,item+itm_idx_len) matX_sample_score(user,item+itm_idx_len); ...
                      matX_sample_val(user,item+2*itm_idx_len) matX_sample_score(user,item+2*itm_idx_len); matX_sample_val(user,item+3*itm_idx_len) matX_sample_score(user,item+3*itm_idx_len); ...
                      matX_sample_val(user,item+4*itm_idx_len) matX_sample_score(user,item+4*itm_idx_len)])

                index = 1;
                range = min(N, 100);
                if is_SN == true
                    plot(full([G_matBeta(1:range,:)*G_matTheta(index,:)' D_matBeta(1:range,:)*D_matTheta(index,:)'/sp_norm_theta/sp_norm_beta matX_train(index,1:range)' matX_test(index,1:range)' G_matX_tilde(index,1:range)']));
                else
                    plot(full([G_matBeta(1:range,:)*G_matTheta(index,:)' D_matBeta(1:range,:)*D_matTheta(index,:)' matX_train(index,1:range)' matX_test(index,1:range)' G_matX_tilde(index,1:range)']));
                end
            end

            if itr >= MaxItr
                IsConverge = true;
            end
        end
        
        
        %% Select the best result according to the performance on the validation set
        e = (cputime - t) / MaxItr;
        fprintf('computational time per iteration: %f\n', e);
        
        QQQ = sum(valid_precision(:,1) + valid_precision(:,2), 2);
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








