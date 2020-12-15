function [prec, rec, ndcg, mrr] = Validation(valid_type, DATA, matTheta, matBeta, topK, usr_idx, usr_idx_len, itm_idx_len)
    
    global N                 % number of items
    global matX_train        % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation

    if strcmp(valid_type, 'validation')
        if usr_idx_len > 5000 && itm_idx_len > 20000
            user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
        else
            user_probe = usr_idx;
        end
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos') || strcmp(DATA, 'MovieLens20M') || strcmp(DATA, 'Jester2')
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_valid(user_probe,:), matX_train(user_probe,:), matTheta(user_probe,:), matBeta, topK);
        else
            [is_X_valid_binary, js_X_valid_binary, vs_X_valid_binary] = find(matX_valid(user_probe,:));
            matX_valid_binary = sparse(is_X_valid_binary, js_X_valid_binary, ones(length(vs_X_valid_binary), 1), length(user_probe), N);
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_valid_binary, matX_train(user_probe,:), matTheta(user_probe,:), matBeta, topK);
        end
        fprintf('validation precision: %f\n', prec(1));
        fprintf('validation recall: %f\n', rec(1));
        fprintf('validation nDCG: %f\n', ndcg(1));
    elseif strcmp(valid_type, 'probing')
        if usr_idx_len > 5000 && itm_idx_len > 20000
            user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
        else
            user_probe = usr_idx;
        end
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos') || strcmp(DATA, 'MovieLens20M') || strcmp(DATA, 'Jester2')
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), matTheta(user_probe,:), matBeta, topK);
        else
            [is_X_test_binary, js_X_test_binary, vs_X_test_binary] = find(matX_test(user_probe,:)+matX_valid(user_probe,:));
            matX_test_binary = sparse(is_X_test_binary, js_X_test_binary, ones(length(vs_X_test_binary), 1), length(user_probe), N);
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_test_binary, matX_train(user_probe,:), matTheta(user_probe,:), matBeta, topK);
        end
        fprintf('testing precision: %f\n', prec(1));
        fprintf('testing recall: %f\n', rec(1));
        fprintf('testing nDCG: %f\n', ndcg(1));
    else
        if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos') || strcmp(DATA, 'MovieLens20M') || strcmp(DATA, 'Jester2')
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), matTheta(usr_idx,:), matBeta, topK);  
        else
            [is_X_test_binary, js_X_test_binary, vs_X_test_binary] = find(matX_test(usr_idx,:)+matX_valid(usr_idx,:));
            matX_test_binary = sparse(is_X_test_binary, js_X_test_binary, ones(length(vs_X_test_binary), 1), length(usr_idx), N);
            [prec, rec, ndcg, mrr] = Evaluate_ALL(matX_test_binary, matX_train(usr_idx,:), matTheta(usr_idx,:), matBeta, topK);
        end
        fprintf('total testing precision: %f\n', prec);
        fprintf('total testing recall: %f\n', rec);
        fprintf('total testing nDCG: %f\n', ndcg);
    end
end

