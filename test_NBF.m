
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

% ----- ExpoHPF -----
global matTheta        % dim(M, K): latent document-topic intensities
global matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
global matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

global matBeta         % dim(N, K): latent word-topic intensities
global matBeta_Shp     % dim(N, K): varational param of matBeta (shape)
global matBeta_Rte     % dim(N, K): varational param of matBeta (rate)

global matEpsilon      % dim(M, 1): latent word-topic intensities
global matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
global matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

global matEta          % dim(N, 1): latent word-topic intensities
global matEta_Shp      % dim(N, 1): varational param of matEta (shape)
global matEta_Rte      % dim(N, 1): varational param of matEta (rate)

global matGamma        % dim(M, 1): latent word-topic intensities
global matGamma_Shp    % dim(M, 1): varational param of matEpsilonX (shape)
global matGamma_Rte    % dim(M, 1): varational param of matEpsilonX (rate)

global matDelta        % dim(N, 1): latent word-topic intensities
global matDelta_Shp    % dim(N, 1): varational param of matEtaX (shape)
global matDelta_Rte    % dim(N, 1): varational param of matEtaX (rate)

global vec_matR_ui
global vec_matD_ui

global prior
global matD

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall



%% Experimental Settings

env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', ...
           'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm1K', 'LastFm2K', 'LastFm360K', 'LastFm360K_2K', 'EchoNest',  ...
           'ML100KPos', 'Jester2', 'ModCloth', 'EachMovie'};
MODEL = "NBF";
ENV = env_type{1};
DATA = dataset{15};

NUM_RUNS = 5;
check_step = 10;

ini_scale = 0.001;
Ks = [20];
topK = [5, 10, 15, 20, 50];
switch DATA
    case 'SmallToy'
        MaxItr = 800;
        Ks = [4];
        topK = [1, 2, 3, 5];
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'SmallToyML'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'ML50'
        MaxItr = 400;
        Ks = [4];
        topK = [1, 2, 3, 5];
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'MovieLens100K'
        MaxItr = 1200;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'MovieLens1M'
        MaxItr = 400;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'MovieLens20M'
        MaxItr = 400;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'Jester2'
        MaxItr = 400;
        Ks = [20];
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'ModCloth'
        MaxItr = 6;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'EachMovie'
        MaxItr = 400;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'LastFm2K'
        MaxItr = 400;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'LastFm1K'
        MaxItr = 400;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'LastFm360K'
        MaxItr = 1200;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   200, 1e6];
    case 'LastFm360K_2K'
        MaxItr = 1200;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
    case 'ML100KPos'
        MaxItr = 1200;
        prior = [0.3, 0.3, 0.3, ...
                   0.3, 0.3, 0.3, ...
                   2, 1e3];
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

if strcmp(MODEL, 'BHPF')
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    matX_train = sparse(is_X_train, js_X_train, ones(length(vs_X_train), 1), M, N);
end


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
        
        % Sample usr_idx, itm_idx
        [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);
        
        itr = 0;
        IsConverge = false;
        avgGamma = zeros(MaxItr, 1);
        total_time = 0;
        while IsConverge == false
            t = cputime;
            itr = itr + 1;
            fprintf('Run: %d , Itr: %d  K = %d  ==> ', num, itr, K);
            fprintf('subPredict_X: ( %d , %d ) , nnz = %d\n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx))); 

            %% Train generator G
            %
            % Train generator G given samples and their scores evluated by D
            Learn_NBF();
            
            total_time = total_time + (cputime - t);

            %% Validation & Testing
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

            avgGamma(itr,1) = mean(mean(matGamma));
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
        
        fprintf('%s: MaxItr / Itr = %d / %d\n', MODEL, MaxItr, itr);
        fprintf('Computing time per epoch : %f sec\n\n', total_time / itr);
    end
end


save matPrecNRecall











