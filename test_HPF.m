
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
model_type = {'HPF', 'BHPF'};
env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', 'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm2K', 'LastFm1K', 'EchoNest', 'LastFm360K_2K', 'LastFm360K', ...
           'ML100KPos', 'BX', 'Jester2', 'ModCloth', 'EachMovie'};
MODEL = model_type{1};
ENV = env_type{1};
DATA = 'EachMovie';

NUM_RUNS = 5;
likelihood_step = 10000;
check_step = 50;

switch DATA
    case 'SmallToy'
        [ M, N ] = LoadUtilities('/home/ian/Dataset/SmallToy_train.csv', '/home/ian/Dataset/SmallToy_test.csv', '/home/ian/Dataset/SmallToy_valid.csv');
    case 'SmallToyML'
        Ks = [4];
        topK = [5, 10, 15, 20];
    case 'MovieLens100K'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 200;
    case 'MovieLens1M'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 200;
    case 'MovieLens20M'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 200;
    case 'LastFm2K'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 400;
    case 'LastFm1K'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 400;
    case 'LastFm360K'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 100;
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
    case 'LastFm360K_2K'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 100;
    case 'ML100KPos'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
    case 'BX'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 1000;
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
    case 'Jester2'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 100;
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
    case 'ModCloth'
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        check_step = 1;
        MaxItr = 10;
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
    case 'EachMvoie'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 200;
    case 'EchoNest'
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        Ks = [20];
        topK = [5, 10, 15, 20, 50];
        MaxItr = 600;
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
        
        % Sample usr_idx, itm_idx
        usr_idx = 1:M;
        itm_idx = 1:N;
        usr_idx(sum(matX_train(usr_idx,:),2)==0) = [];
        itm_idx(sum(matX_train(:,itm_idx),1)==0) = [];
        usr_idx_len = length(usr_idx);
        itm_idx_len = length(itm_idx);
        
        total_time = 0;
        while IsConverge == false
            itr = itr + 1;
            
            if mod(itr, 5) == 0 
                fprintf('Run: %d , Itr: %d  K = %d  ==> ', num, itr, K);
                fprintf('subPredict_X: ( %d , %d ) , nnz = %d \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)));
            end
            
            %% Train HPF
            timer = tic; 
            
            Learn_HPF();
            
            total_time = total_time + toc(timer);
            
            %% Validation & Testing
            if likelihood_step > 0 && mod(itr, likelihood_step) == 0
                tmpX = sum(HPF_matTheta(is_X_train,:) .* HPF_matBeta(js_X_train,:), 2);
                obsrv_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX);
                loglikelihood(itr / likelihood_step, 1) = mean(obsrv_poisson);
                
                fprintf('Log likelihood of Poisson: %f\n', mean(obsrv_poisson));
            end
            
            if check_step > 0 && mod(itr, check_step) == 0
                indx = itr / check_step;
                
                % Calculate the metrics on validation set
                [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Validation('validation', DATA, HPF_matTheta, HPF_matBeta, ...
                                                                                                                    topK, usr_idx, usr_idx_len, itm_idx_len);
                
                % Calculate the metrics on testing set
                [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Validation('probing', DATA, HPF_matTheta, HPF_matBeta, ...
                                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
                                                                                                            
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
