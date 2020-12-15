
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta
 
% ----- HNBF -----
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

global vecMu           % dim(M, 1): approximate matD
global vecMu_Shp       % dim(M, 1): approximate matD
global vecMu_Rte       % dim(M, 1): approximate matD
global matGamma        % dim(M, K): approximate matD
global matGamma_Shp    % dim(M, K): approximate matD
global matGamma_Rte    % dim(M, K): approximate matD

global vecPi           % dim(N, 1): approximate matD
global vecPi_Shp       % dim(N, 1): approximate matD
global vecPi_Rte       % dim(N, 1): approximate matD
global matDelta        % dim(N, K): approximate matD
global matDelta_Shp    % dim(N, K): approximate matD
global matDelta_Rte    % dim(N, K): approximate matD

global vec_matR_ui_shp
global vec_matR_ui_rte
global vec_matR_ui
global vec_matD_ui_shp
global vec_matD_ui_rte
global vec_matD_ui

global prior

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall



%% Experimental Settings
%
% Statistics of Datasets
% -------------------------------------------------------------------------
% 1. MovieLens100K =>  M = 943     , N = 1682   , NNZ = 100K
% 2. MovieLens1M   =>  M = 6040    , N = 3900   , NNZ = 1M
%    MovieLens20M  =>  M = 138493 ,  N = 25854  , NNZ = 20M
% 3. LastFm2K      =>  M = 1892    , N = 17632  , NNX = 92,834
% 4. LastFm1K      =>  MSetting = 992     , N = 174091 , NNZ = 898K
% 5. LastFm360K_2K =>  M = 2000    , N = 1682   , NNZ = 
% 6. LastFm360K    =>  M = 359349  , N = 292589 , NNZ = 17,559,486
% 7. ML100KPos     =>  M = 943     , N = 1682   , NNZ = 67,331

model_type = {'HNBF', 'FactorHNBF', 'FastHNBF'};
env_type = {'OSX', 'Linux'};
dataset = {'SmallToy', 'SmallToyML', 'ML50', ...
           'MovieLens100K', 'MovieLens1M', 'MovieLens20M', ...
           'LastFm1K', 'LastFm2K', 'LastFm360K', 'LastFm360K_2K', 'EchoNest',  ...
           'ML100KPos', 'Jester2', 'ModCloth', 'EachMovie'};

%
% Set Configuration
% -------------------------------------------------------------------------
MODEL = model_type{3};
ENV = env_type{1};
DATA = 'MovieLens1M';

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


%% 
% Experiments
% -------------------------------------------------------------------------

for kk = 1:length(Ks)
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
        

        % Initialize the model
        newHNBFs(MODEL, ini_scale, usr_zeros, itm_zeros);
         
        
        % Sample usr_idx, itm_idx
        [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);
        
        itr = 0;
        last_poisson_likeli = -1e10;
        IsConverge = false;
        total_time = 0;
        while IsConverge == false        
            itr = itr + 1;
            lr = 1.0;

            if mod(itr, 5) == 0 
                fprintf('Run: %d , Itr: %d  K = %d  ==> ', num, itr, K);
                fprintf('subPredict_X: ( %d , %d ) , nnz = %d , lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), lr);
            end


            %% Train generator G
            %
            timer = tic;
            % Train generator G given samples and their scores evluated by D
            if strcmp(MODEL, 'HNBF')
                Learn_HNBF(lr);
            elseif strcmp(MODEL, 'FactorHNBF')
                Learn_FactorHNBF(lr);
            elseif strcmp(MODEL, 'FastHNBF')
                Learn_FastHNBF(lr);
            else
            end
 
            total_time = total_time + toc(timer);

            %% Calculate log likelihood of Poisson and Negative Binomial
            %
            if likelihood_step > 0 && mod(itr, likelihood_step) == 0
                
                tmpX = sum(matTheta(is_X_train,:) .* matBeta(js_X_train,:), 2);
                tmpXX = vec_matD_ui .* tmpX;
                tmpSparse = sum(sum(matGamma).*sum(matDelta), 2)/M/N;
                vstrain_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX);
                vstrain_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpXX);
                vecD_tmpX(itr / likelihood_step, 1) = mean(vec_matD_ui);
                vecD_tmpX(itr / likelihood_step, 2) = mean(tmpX);
                vecD_tmpX(itr / likelihood_step, 3) = mean(tmpSparse);
                
                [is_X_test, js_X_test, vs_X_test] = find(matX_test);
                tmpX = sum(matTheta(is_X_test,:) .* matBeta(js_X_test,:), 2);
                tmpXX = sum(matGamma(is_X_test,:) .* matDelta(js_X_test,:), 2) .* tmpX;
                vstest_poisson = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpX);
                vstest_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpXX);
                
                [is_X_valid, js_X_valid, vs_X_valid] = find(matX_valid);
                tmpX = sum(matTheta(is_X_valid,:) .* matBeta(js_X_valid,:), 2);
                tmpXX = sum(matGamma(is_X_valid,:) .* matDelta(js_X_valid,:), 2) .* tmpX;
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
            end
            
            
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
                
                % Draw a consumption sample 
                range = min(N, 100);
                index = 30;
                tmp1 = matBeta(1:range,:) * matTheta(index,:)';
                tmp2 = matDelta(1:range,:) * matGamma(index,:)' / K;
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
            if mod(itr, likelihood_step) == 0
                if itr > base_steps && mean(vstrain_poisson) - last_poisson_likeli < stop_criteria
                    IsConverge = true;
                    break
                end
                last_poisson_likeli = mean(vstrain_poisson);
                Best_matTheta = matTheta;
                Best_matBeta = matBeta;
            end
        end
        
        [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Validation('testing', DATA, Best_matTheta, Best_matBeta, ...
                                                                                                topK, usr_idx, usr_idx_len, itm_idx_len);
          
        % Record the experimental result
        matPrecNRecall((kk-1)*NUM_RUNS+num,1:length(topK)*4) = [total_test_precision total_test_recall total_test_nDCG total_test_MRR];
        matPrecNRecall((kk-1)*NUM_RUNS+num,length(topK)*4+1:end) = [valid_precision(end,:) valid_recall(end,:) valid_nDCG(end,:) valid_MRR(end,:)];
        
        fprintf('%s: MaxItr / Itr = %d / %d, StopCriteria = %f\n', MODEL, MaxItr, itr, stop_criteria);
        fprintf('Computing time per epoch : %f sec\n\n', total_time / itr);
    end
end

  
save matPrecNRecall
