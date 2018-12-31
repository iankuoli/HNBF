function [precision, recall] = EvaluateBPMF_PrecNRec(matX_test, matX_test_train, matTheta, matBeta, topK, mean_rating)
    [vec_usr_idx, j, v] = find(sum(matX_test, 2));
    list_vecPrecision = zeros(1, length(topK));
    list_vecRecall = zeros(1, length(topK));
    log_likelihood = 0;
    step_size = 300;
    denominator = 0;

    for j = 1:ceil(length(vec_usr_idx)/step_size)
        batch_step = (1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx));

        if isempty(batch_step)
          break
        end

        % Compute the Precision and Recall
        matPredict = matTheta(vec_usr_idx(batch_step),:) * matBeta' + mean_rating;
        matPredict = matPredict - matPredict .* (matX_test_train(vec_usr_idx(batch_step), :) > 0);
        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(vec_usr_idx(batch_step), :), matPredict, topK);
                
        % ----- single processor -----
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
        list_vecRecall = list_vecRecall + sum(vec_recall, 1);
        denominator = denominator + length(batch_step);
    end
    precision = list_vecPrecision / denominator;
    recall = list_vecRecall / denominator;
    
end