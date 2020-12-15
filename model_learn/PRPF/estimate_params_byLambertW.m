function solution_xui_xuj = estimate_params_byLambertW(l_function_s, h_function_s, vec_predict_X_u, vec_s, js)
    W_tmp = -l_function_s .* exp(h_function_s);
    W_toosmall_mask = W_tmp <= -1/exp(1);
    W_toolarge_mask = W_tmp > 10e+30;
    W_mask = (ones(length(W_tmp),1) - W_toosmall_mask - W_toolarge_mask) > 0;

    vec_lambda = zeros(length(js), 2);
    vec_lambda(W_mask, :) = bsxfun(@times, [Lambert_W(W_tmp(W_mask), 0), Lambert_W(W_tmp(W_mask), -1)], -1 ./ l_function_s(W_mask));
    vec_lambda(W_toolarge_mask,:) = -repmat((h_function_s(W_toolarge_mask)) ./ l_function_s(W_toolarge_mask), 1, 2);

    [v_better, i_better] = min(abs(bsxfun(@plus, vec_lambda, - vec_s)), [], 2);
    mask_better = sparse(1:length(js), i_better, ones(length(js), 1), length(js), 2);
    vec_lambda(isnan(vec_lambda)) = 0;

    solution_xui_xuj = sum(vec_lambda .* mask_better, 2)';
    solution_xui_xuj(isnan(solution_xui_xuj)) = vec_predict_X_u(isnan(solution_xui_xuj));
    solution_xui_xuj(solution_xui_xuj==inf) = 1;
end

