
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

% ----- BPR -----
global NMF_matTheta          % dim(M, K): latent document-topic intensities
global NMF_matBeta           % dim(N, K): latent word-topic intensities


global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation


global matPrecNRecall



%% Experimental Settings
model_type = {'NMF', 'BNMF'};
env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', ...
           'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm1K', 'LastFm2K', 'LastFm360K', 'LastFm360K_2K', 'EchoNest',  ...
           'ML100KPos', 'Jester2', 'ModCloth', 'EachMovie'};

%
% Set Configuration
% -------------------------------------------------------------------------
MODEL = 'BNMF';
ENV = env_type{1};
DATA = 'LastFm1K';

NUM_RUNS = 10;
likelihood_step = 10;
check_step = 10;
base_steps = 300;

ini_scale = 0.001;
Ks = [20];
topK = [5, 10, 15, 20, 50];

[MaxItr, prior, stop_criteria] = config_HNBFs(DATA, MODEL);
if strcmp(DATA, 'ModCloth')
    check_step = 1;
    likelihood_step = 1;
    base_steps = 3;
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

if strcmp(MODEL, 'BNMF')
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    matX_train = sparse(is_X_train, js_X_train, ones(length(vs_X_train), 1), M, N);
end


%% Experiments

for kk = 1:length(Ks)
    K = Ks(kk);
    
    for num = 1:NUM_RUNS
        %% Paramter setting
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
        
        [is_X_train, js_X_train, vs_X_train] = find(matX_train);
        [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);
        
        
        %% Training
        %
        [NMF_matTheta, NMF_matBeta] = nnmf(matX_train, K);
        NMF_matBeta = NMF_matBeta';
    
        
        
        %% Testing
        %
        [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Validation('testing', DATA, NMF_matTheta, NMF_matBeta, ...
                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
          
        % Record the experimental result
        matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [total_test_precision total_test_recall total_test_nDCG total_test_MRR];
        matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_nDCG(end,:) valid_MRR(end,:)];
        
        fprintf('%s: MaxItr / Itr = %d / %d = %f\n', MODEL, MaxItr, itr);
        fprintf('Computing time per epoch : %f sec\n\n', total_time / itr);
        
    end
end


save matPrecNRecall











