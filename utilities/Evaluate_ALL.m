function [precision, recall, nDCG, MRR] = Evaluate_ALL(matX_test, matX_test_train, matTheta, matBeta, topK)
    [vec_usr_idx, j, v] = find(sum(matX_test, 2));
    list_vecPrecision = zeros(1, length(topK));
    list_vecRecall = zeros(1, length(topK));
    list_vecNDCG = zeros(1, length(topK));
    list_vecMRR = zeros(1, length(topK));
    step_size = 300;
    denominator = 0;

    for j = 1:ceil(length(vec_usr_idx)/step_size)
        batch_step = (1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx));

        if isempty(batch_step)
          break
        end

        % Compute the Precision and Recall
        matPredict = matTheta(vec_usr_idx(batch_step),:) * matBeta';
        matPredict = matPredict - matPredict .* (matX_test_train(vec_usr_idx(batch_step), :) > 0);
        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(vec_usr_idx(batch_step), :), matPredict, topK);
        vec_nDCG = MeasureNDCG(matX_test(vec_usr_idx(batch_step), :), matPredict, topK);
        vec_MRR = MeasureMRR(matX_test(vec_usr_idx(batch_step), :), matPredict, topK);
        
        % ----- single processor -----
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
        list_vecRecall = list_vecRecall + sum(vec_recall, 1);
        list_vecNDCG = list_vecNDCG + sum(vec_nDCG, 1);
        list_vecMRR = list_vecMRR + sum(vec_MRR, 1);
        denominator = denominator + length(batch_step);
    end
    precision = list_vecPrecision / denominator;
    recall = list_vecRecall / denominator;
    nDCG = list_vecNDCG / denominator;
    MRR = list_vecMRR / denominator;
end