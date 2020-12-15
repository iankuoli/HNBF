function predict_X = update_matX_predict_bk2(LTR, usr_idx, itm_idx, delta, alpha, SVI_lr, C)

    global matX_predict
    global PRPF_matTheta
    global PRPF_matBeta
    
    global matX_train
    
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
    
    [is_X, js_X, vs_X] = find(matX_train(usr_idx, itm_idx));
    vs_prior_X = sum(PRPF_matTheta(usr_idx(is_X),:) .* PRPF_matBeta(itm_idx(js_X), :), 2);
    prior_X = sparse(is_X, js_X, vs_prior_X, usr_idx_len, itm_idx_len);

    predict_X = matX_predict(usr_idx, itm_idx);
    
    vec_time = zeros(usr_idx_len, 1);
    
    if strcmp(LTR, 'pairPRPF')
        %
        % Pair-wise ranking is based on logistic function.
        % To approximate \hat{x}_{ui} by a Poisson distribution.
        % Use 2-th Taylor series for approximation.
        % Do not use conjugate prior, minimize the difference between the two expectation directly!
        %
        for u = 1:usr_idx_len
            
            tt = cputime;
            
            u_idx = usr_idx(u);
            if nnz(matX_train(u_idx, itm_idx)) < 2
                continue;
            end
            [is, js, vs] = find(matX_train(u_idx, itm_idx));

            priorX_seg = prior_X(u,:)';
            predictX_seg = predict_X(u,:)';

            vec_prior_X_u = full(priorX_seg(js));
            vec_predict_X_u = full(predictX_seg(js));
            vec_matX_u = full(matX_train(u_idx, itm_idx(js)))';

            %% Compute logisitic(\hat{x}_{ui}) for all nonzero x_{ui}
            exp_diff_predict_xij_h = exp(vec_predict_X_u);
            partial_1_diff_predict_xij_h = 1 ./ (1 + exp_diff_predict_xij_h);
            partial_1_diff_predict_xij_h(isnan(partial_1_diff_predict_xij_h)) = 1;
            partial_2_diff_predict_xij_h = -exp_diff_predict_xij_h ./ (1 + exp_diff_predict_xij_h) .^ 2;
            partial_2_diff_predict_xij_h(isnan(partial_2_diff_predict_xij_h)) = 1;

            %
            % Select the optimal s_{ui}
            % Compute partial_1_diff_predict_xij_L, partial_2_diff_predict_xij_L
            %
            mat_diff_matX_u = bsxfun(@plus, vec_matX_u, -vec_matX_u');
            mat_exp_diff_predictX_u = exp(delta * bsxfun(@plus, vec_predict_X_u, -vec_predict_X_u'));
            mat_logistic_diff_predictX_u = 1 ./ (1+mat_exp_diff_predictX_u);
            matL_partial_sui = full(C/length(vec_matX_u) * delta * bsxfun(@plus, mat_logistic_diff_predictX_u * (mat_diff_matX_u~=0), -sum(mat_diff_matX_u>0,1)));
            matL_partial_sui = bsxfun(@plus, matL_partial_sui, alpha * partial_1_diff_predict_xij_h);
            [partial_1_diff_f, min_idx] = min(abs(matL_partial_sui));
            partial_1_diff_f = full(sum(matL_partial_sui .* sparse(min_idx, 1:length(matL_partial_sui), ones(1,length(matL_partial_sui)), length(matL_partial_sui), length(matL_partial_sui))))';

            vec_s = vec_predict_X_u(min_idx);

            matL_partial2_sui = -C/length(vec_matX_u) * delta^2 * (mat_logistic_diff_predictX_u .* mat_exp_diff_predictX_u ./ (1+mat_exp_diff_predictX_u)) * (mat_diff_matX_u~=0);
            partial_2_diff_f = full(sum(matL_partial2_sui .* sparse(min_idx, 1:length(min_idx), ones(length(min_idx),1), length(min_idx), length(min_idx)), 1))';
            partial_2_diff_f = partial_2_diff_f + alpha * partial_2_diff_predict_xij_h(min_idx);

            l_function_s = partial_2_diff_f;              
            h_function_s = (partial_1_diff_f + (0.5 - vec_s) .* l_function_s) + log(vec_prior_X_u);
    
            solution_xui_xuj = estimate_params_byLambertW(l_function_s, h_function_s, vec_predict_X_u, vec_s, js);      

            if any(isnan(solution_xui_xuj))
                fprintf('nan');
            end

            if sum(any(solution_xui_xuj == inf))
                fprintf('inf');
            end

            if sum(any(solution_xui_xuj == -inf))
                fprintf('-inf');
            end
            solution_xui_xuj(solution_xui_xuj<1e-10) = 1e-10;
            predict_X(u, js) = solution_xui_xuj;
                        
            tt2 = cputime;
            vec_time(u,1) = tt2 - tt;
        end
        if size(predict_X, 1) ~= usr_idx_len
            fprintf('FUCKKK');
        end
        if size(predict_X, 2) ~= itm_idx_len
            fprintf('FUCKKK');
        end
    else
        %
        % List-wise ranking is based on logistic function.
        % To approximate \hat{x}_{ui} by a Poisson distribution.
        % Use 2-th Taylor series for approximation.
        % Do not use conjugate prior, minimize the difference between the two expectation directly!
        %
        for u = 1:usr_idx_len
            u_idx = usr_idx(u);
            if nnz(matX_train(u_idx, itm_idx)) < 2
                continue;
            end
            [is, js, vs] = find(matX_train(u_idx, itm_idx));
            js = js';
            
            priorX_seg = prior_X(u,:)';
            predictX_seg = predict_X(u,:)';

            vec_prior_X_u = full(priorX_seg(js));
            vec_predict_X_u = full(predictX_seg(js));
            vec_matX_u = full(matX_train(u_idx, itm_idx(js)))';
                
            [decreasing_matX_u, decreasing_index_matX_u] = sort(vec_matX_u, 'descend');
            num_I_u = length(decreasing_index_matX_u);

            partial_2_diff_f = zeros(num_I_u, 1);
            
            
            sort_predX = vec_predict_X_u(decreasing_index_matX_u);
            if strcmp(LTR, 'list_expPRPF')
                % exponential tranformation
                sort_transX = exp(delta * sort_predX);
            else
                % linear transformation
                sort_transX = delta * sort_predX;
            end
            
            matL_partial_sui = zeros(num_I_u);
            
            % Compute s_j^{\vert \mathcal{I}_u \vert}
            vec_sum_transX = cumsum(sort_transX, 'reverse');
            
            if strcmp(LTR, 'list_expPRPF')
                % exponential tranformation
                for pi_ui = 1:num_I_u
                    for cand = 1:num_I_u
                        sum_h_ui_x = 0;
                        for j = 1:pi_ui
                            h_ui_x = sort_transX(cand) / (vec_sum_transX(j) - sort_transX(pi_ui) + sort_transX(cand));
                            sum_h_ui_x = sum_h_ui_x + h_ui_x;
                        end
                        matL_partial_sui(pi_ui, cand) = delta - ...
                                                        delta * sum_h_ui_x + alpha / (1. + exp(sort_predX(cand)));
                    end
                end
            else
                % linear transformation
                for pi_ui = 1:num_I_u
                    for cand = 1:num_I_u
                        sum_h_ui_x = 0;
                        for j = 1:pi_ui
                            h_ui_x = 1. / (vec_sum_transX(j) - sort_transX(pi_ui) + sort_transX(cand));
                            sum_h_ui_x = sum_h_ui_x + h_ui_x;
                        end
                        matL_partial_sui(pi_ui, cand) = 1 / sort_predX(cand) - ...
                                                        delta * sum_h_ui_x + alpha / (1. + exp(sort_predX(cand)));
                    end
                end
            end
            
            [partial_1_diff_f, min_idx] = min(abs(matL_partial_sui), [], 2);
            
            transform_sui = sort_transX(min_idx);
            sui = sort_predX(min_idx);   
            exp_sui = exp(sui);
            
            if strcmp(LTR, 'list_expPRPF')
                % exponential tranformation
                for pi_ui=1:num_I_u
                    sum_b = 0;
                    for j = 1:pi_ui
                        h_ui_x = transform_sui(pi_ui) ./ (vec_sum_transX(j) - sort_transX(pi_ui) + transform_sui(pi_ui));
                        sum_b = sum_b + h_ui_x * (1. - h_ui_x);
                    end
                    partial_2_diff_f(pi_ui) = - delta^2 * sum_b - ...
                                              alpha * exp_sui(pi_ui) / (1 + exp_sui(pi_ui))^2;
                end
            else
                % linear transformation
                for pi_ui=1:num_I_u
                    sum_b = 0;
                    for j = 1:pi_ui
                        h_ui_x = 1 ./ (vec_sum_transX(j) - sort_transX(pi_ui) + transform_sui(pi_ui));
                        sum_b = sum_b + h_ui_x^2;
                    end
                    partial_2_diff_f(pi_ui) = -1./sui(pi_ui)^2 + ...
                                              delta^2 * sum_b - ...
                                              alpha * exp_sui(pi_ui) / (1. + exp_sui(pi_ui))^2;
                end
            end
            
            %
            % Compute function l and h
            % 
            vec_s = vec_predict_X_u(decreasing_index_matX_u(min_idx));
            vec_prior = vec_prior_X_u(decreasing_index_matX_u(min_idx));
            l_function_s = partial_2_diff_f;
            h_function_s = partial_1_diff_f + (0.5 - vec_s) .* l_function_s + log(vec_prior);

            %
            % Estimate \tilde{x}_{ui} approximately by Lamber W function
            %
            solution_xui_xuj = estimate_params_byLambertW(l_function_s, h_function_s, vec_predict_X_u, vec_s, js);

            %
            % So far, the indices of partial_1_diff_f and partial_2_diff_f follow decreasing order.
            % Now we will map the sorted index to the original index.
            %
            qqq = full(sparse(decreasing_index_matX_u, 1, 1:num_I_u, num_I_u, 1));
            solution_xui_xuj = solution_xui_xuj(qqq);

            if any(isnan(solution_xui_xuj))
                fprintf('nan');
            end

            if sum(any(solution_xui_xuj == inf))
                fprintf('inf');
            end

            if sum(any(solution_xui_xuj == -inf))
                fprintf('-inf');
            end

            solution_xui_xuj(solution_xui_xuj<1e-10) = 1e-10;
            predict_X(u, js) = solution_xui_xuj;
        end
    end

    matX_predict(usr_idx, itm_idx) = (1-SVI_lr) * matX_predict(usr_idx, itm_idx) + SVI_lr * predict_X;
end

