function [val_sigma] = G_Learn_BNMF_userwise(lr_BBVI, usr_idx, itm_idx, S, R, tau, user_weights, tauQ, m, val_sigma, varargin)

    % Usage
    %   [] = BNMF_Learn(usr_idx, itm_idx, m, [options])
    %
    % Input -------------------------------
    %   usr_idx   Number of users
    %   itm_idx   Number of items
    %   m         Index of Gibbs sampling times
    %   options -------------------------------
    %     reg       Regularization type ('pp', 'vol', or 'dist')
    %     gamma     Volume prior, hyperparameter of regularization term J_w
    %     theta     Prior for val_sigma
    %     sig_k     Prior for val_sigma
    %     skip      Number of initial samples to skip (default 100)
    %     stride    Return every n'th sample (default 1)
    %
    %
    
    global is_useGAN
    
    global K
    
    global G_matTheta
    global G_matBeta
    global G_kappa
    
    global G_matX_tilde
    
    global matX_train
    global matSample_val
    global matSample_Score
    global vecSample_i
    global vecSample_j
    
    
    opts      = mgetopt(varargin);
    reg       = mgetopt(opts, 'reg', 'pp');   % Regularization type ('pp', 'vol', or 'dist')
    gamma     = mgetopt(opts, 'gamma', 0);    % Hyperparameter of regularization term J_w
    theta     = mgetopt(opts, 'theta', 0);    % Prior for val_sigma
    sig_k     = mgetopt(opts, 'k', 0);        % Prior for val_sigma
    skip      = mgetopt(opts, 'skip', 0);     % Number of initial samples to skip (default 100)
    stride    = mgetopt(opts, 'stride', 1);   % Return every n'th sample (default 1)
    rep       = mgetopt(opts, 'rep', 1);      % Repeating time per sample (default 1)
    
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
    
    [is_X_tilde, js_X_tilde, vs_X_tilde] = find(G_matX_tilde(usr_idx, itm_idx));
    nnz_matX_train = nnz(matX_train(usr_idx, itm_idx));
    
    sample_indx_size = length(vecSample_i);
    
    % Compute the user-wise normalized weight.
    elem_weight = user_weights(vecSample_i(1:nnz_matX_train));
    
    
    %% ---- Update G_matX_tilde ---- %
    if is_useGAN == true
        mean_vs_tilde_seeds = mean(vs_X_tilde);
        mean_ThetaBeta = mean(sum(G_matTheta(usr_idx(is_X_tilde),:) .* G_matBeta(itm_idx(js_X_tilde),:), 2));

        % Compute prior
        vs_X_prior = sum(G_matTheta(usr_idx(vecSample_i),:) .* G_matBeta(itm_idx(vecSample_j),:), 2);

        % Compute sampling seeds
        vs_smpl_seeds = [vs_X_tilde; vs_X_prior((nnz_matX_train+1):end) + (mean_vs_tilde_seeds - mean_ThetaBeta)];

        % Sample nonzero & zero data from vs_X_tilde and prior
        samples_vs_X_seeds = zeros(sample_indx_size, S);
        for s = 1:S
            tmp = vs_smpl_seeds + sqrt(val_sigma) * rand(length(vs_smpl_seeds), 1);
            tmp(tmp < 0.1) = 0.1;
            samples_vs_X_seeds(:, s) = tmp;
        end

        log_q_X_train = -bsxfun(@plus, samples_vs_X_seeds, -vs_smpl_seeds).^2 ./ (2 * val_sigma);

        % Compute the gradient of G_matX_tilde
        log_r_elem_real = -(matSample_val(1:nnz_matX_train,:) - repmat(vs_X_tilde, 1, R)).^2 ./ (2 * 1/val_sigma);
        J_original = mean(mean(log_r_elem_real .* matSample_Score(1:nnz_matX_train,:), 2) .* elem_weight);

        grad_G_vecX_seeds = zeros(sample_indx_size, 1);
        for s = 1:S
            % Compute the objective, i.e., J
            log_r_elem_sample = -bsxfun(@plus, matSample_val, -samples_vs_X_seeds(:,s)) .^ 2 ./ (2 * val_sigma);
            J_obj = mean(log_r_elem_sample .* matSample_Score, 2);
            J_obj(1:nnz_matX_train) = J_obj(1:nnz_matX_train) .* elem_weight;
            J_obj(nnz_matX_train+1:end) = J_obj(nnz_matX_train+1:end) * tauQ;

            J_prior = -(samples_vs_X_seeds(:,s) - vs_X_prior) .^ 2 ./ (2 * val_sigma);
            J_prior(nnz_matX_train+1:end) = tau * (usr_idx_len * itm_idx_len - nnz_matX_train) / (sample_indx_size - nnz_matX_train) * J_prior(nnz_matX_train+1:end);
            J_q = log_q_X_train(:, s);
            J = G_kappa * J_obj + J_prior - J_q;

            % Compute the gradient of q
            grad_log_q_X_seeds = 1 / val_sigma * (samples_vs_X_seeds(:,s) - vs_smpl_seeds);

            % Compute grad_G_vecX_tilde
            grad_G_vecX_seeds = grad_G_vecX_seeds + grad_log_q_X_seeds .* J;
        end
        vs_tilde_seeds = vs_smpl_seeds(1:nnz_matX_train) + lr_BBVI * grad_G_vecX_seeds(1:nnz_matX_train) / S;
        vs_zero_seeds = vs_smpl_seeds(nnz_matX_train+1:end) + lr_BBVI * grad_G_vecX_seeds(nnz_matX_train+1:end) / S;

        vs_tilde_seeds(vs_tilde_seeds < 1) = 1;
        vs_zero_seeds(vs_zero_seeds < 0.01) = 0.01;

        rtmp = (usr_idx_len * itm_idx_len - nnz_matX_train) / (sample_indx_size - nnz_matX_train);
        weight = tau * rtmp;
        vs_smpl_seeds = [vs_tilde_seeds; weight * vs_zero_seeds];

        G_matX_tilde(usr_idx, itm_idx) = sparse(is_X_tilde, js_X_tilde, vs_tilde_seeds, usr_idx_len, itm_idx_len);
        G_matX_tilde_tmp = sparse(vecSample_i, vecSample_j, vs_smpl_seeds, usr_idx_len, itm_idx_len);
        if any(isreal(G_matX_tilde_tmp)) == 0
            error('Not a real number.');
        end
        x = sum(vs_smpl_seeds .^ 2) / 2;

        fprintf('   J^G = %f ======> grad_matX_tilde: %f\n', J_original / nnz(matX_train(usr_idx, itm_idx)), mean(mean(vs_smpl_seeds.^2)));
    else
        G_matX_tilde_tmp = matX_train(usr_idx, itm_idx);
        [is_X_train, js_X_train, vs_X_train] = find(matX_train(usr_idx, itm_idx));
        x = sum(vs_X_train .^ 2) / 2;
    end
    
    
    %% ---- Update BNMF ---- %
    
    for t = 1:max(skip * (m==1) + stride * (m>1), 1)
        matC = G_matBeta(itm_idx,:)' * G_matBeta(itm_idx,:);      % matC(k,k) = (\bf{h}_{k:} \bf{h}_{k:}^\top) in Eq. (35)
        matD = G_matX_tilde_tmp * G_matBeta(itm_idx, :);          % the first term in Eq. (37)
        if any(isreal(matD)) == 0
            error('Not a real number.');
        end
        for irep = 1:rep
            for k = 1:K
                kk = [1:k-1 k+1:K];
                switch reg
                    case 'pp'
                      matGt = G_matTheta(usr_idx,kk)' * G_matTheta(usr_idx,kk);
                      adjG = adj(matGt);
                      detG = det(matGt);
                    case 'vol'
                      kp = mod(k, K) + 1;
                      kkpt = mod((1:K-2)+k, K) + 1;
                      matGt = bsxfun(@minus, G_matTheta(usr_idx,kkpt), G_matTheta(usr_idx,kp))' * ...
                              bsxfun(@minus, G_matTheta(usr_idx,kkpt), G_matTheta(usr_idx,kp));
                      adjGt = adj(matGt);
                      detGt = det(matGt);
                    case 'dist'
                end
                for u = 1:usr_idx_len
                    nu = [1:u-1 u+1:usr_idx_len];
                    m1 = (matD(usr_idx(u),k) - G_matTheta(usr_idx(u),kk) * matC(kk,k)) / matC(k,k);  % in Eq. (37)
                    iv1 = matC(k,k) / val_sigma;           % (\bf{h}_{k:} \bf{h}_{k:}^\top)\sigma^{-2} in Eq. (35)
                    switch reg
                        case 'pp'
                            vec_b  = G_matTheta(usr_idx(u), kk)';
                            vec_c  = G_matTheta(usr_idx(nu), kk)' * G_matTheta(usr_idx(nu), k);
                            t2 = gamma * (detG - vec_b' * adjG * vec_b);
                            t1 = -2 * gamma * vec_b'* adjG * vec_c;
                        case 'vol'
                            vec_b  = G_matTheta(usr_idx(u),kkpt)' - G_matTheta(usr_idx(u),kp);
                            vec_c  = (bsxfun(@minus, G_matTheta(usr_idx(nu),kkpt), G_matTheta(usr_idx(nu),kp)))' * ...
                                     (G_matTheta(usr_idx(nu),k) - G_matTheta(usr_idx(nu),kp));
                            t2 = gamma * (detGt - vec_b' * adjGt * vec_b);
                            t1 = -2 * G_matTheta(usr_idx(u),kp) * t2 - 2 * gamma * vec_b' * adjGt * vec_c;
                        case 'dist'
                            t2 = gamma * (K-1) / K;   
                            t1 = -2 * gamma / K * sum(G_matTheta(usr_idx(u), kk));
                        otherwise
                            error('Wrong regularization specified.');
                    end
                    iv2    = t2;                        % s_{mk}^{-2}           in Eq. (45)
                    iv2m2  = -t1 / 2;                   % m_{mk} s_{mk}^{-2}    in Eq. (36)
                    v      = 1 / (iv1 + iv2);           % \line{\sigma}_{mk}^2  in Eq. (35) 
                    mu     = v * (iv1 * m1 + iv2m2);    % \line{\mu}_{mk}       in Eq. (36)
                    s      = sqrt(-2 * (log(rand) - 0.5 * G_matTheta(usr_idx(u),k) .^ 2));
                    ll     = max(-s, -mu / sqrt(v));
                    uu     = s;
                    G_matTheta(usr_idx(u),k) = (rand .* (uu-ll) + ll) * sqrt(v) + mu;
                    if isreal(G_matTheta(usr_idx(u),k)) == 0
                        error('Not a real number.');
                    end
                end
            end
        end
        val_sigma = 1 / gamrnd(usr_idx_len * itm_idx_len / 2 + 1 + sig_k, ...
                               1/(x + theta + sum(sum(G_matTheta(usr_idx,:) .* (G_matTheta(usr_idx,:) * matC - 2 * matD))) / 2));
        
        matE  = G_matTheta(usr_idx,:)' * G_matTheta(usr_idx,:);
        matF  = G_matTheta(usr_idx,:)' * G_matX_tilde_tmp;
        matMu = matE \ matF; 
        matS = matE \ (val_sigma * eye(K));
        G_matBeta(itm_idx, :) = randcg(matMu, matS, eye(K), zeros(K,1), ones(1,K), 1, G_matBeta(itm_idx,:)', rep)';
    end
end



%--------------------------------------------------------------------------
function out = mgetopt(varargin)
    % MGETOPT Parser for optional arguments
    %
    % Usage
    %   Get a parameter structure from 'varargin'
    %     opts = mgetopt(varargin);
    %
    %   Get and parse a parameter:
    %     var = mgetopt(opts, varname, default);
    %        opts:    parameter structure
    %        varname: name of variable
    %        default: default value if variable is not set
    %
    %     var = mgetopt(opts, varname, default, command, argument);
    %        command, argument:
    %          String in set:
    %          'instrset', {'str1', 'str2', ... }
    %
    % Example
    %    function y = myfun(x, varargin)
    %    ...
    %    opts = mgetopt(varargin);
    %    parm1 = mgetopt(opts, 'parm1', 0)
    %    ...

    % Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk

    if nargin==1
        if isempty(varargin{1})
            out = struct;
        elseif isstruct(varargin{1})
            out = varargin{1}{:};
        elseif isstruct(varargin{1}{1})
            out = varargin{1}{1};
        else
            out = cell2struct(varargin{1}(2:2:end), varargin{1}(1:2:end), 2);
        end
    elseif nargin>=3
        opts    = varargin{1};
        varname = varargin{2};
        default = varargin{3};
        validation = varargin(4:end);
        if isfield(opts, varname)
            out = opts.(varname);
        else
            out = default;
        end

        for narg = 1:2:length(validation)
            cmd = validation{narg};
            arg = validation{narg+1};
            switch cmd
                case 'instrset'
                    if ~any(strcmp(arg, out))
                      fprintf(['Wrong argument %sigma = ''%sigma'' - ', ...
                        'Using default : %sigma = ''%sigma''\n'], ...
                        varname, out, varname, default);
                      out = default;
                    end
                case 'dim'
                    if ~all(size(out)==arg)
                      fprintf(['Wrong argument dimension: %sigma - ', ...
                        'Using default.\n'], ...
                        varname);
                      out = default;
                    end
                otherwise
                    error('Wrong option: %sigma.', cmd);
            end
        end
    end
end


function x = randcg(mx, matSxx, matA, vec_b, matAeq, vec_beq, x0, T)
    % RANDCG Random numbers from constrained Gaussian density
    %   p(x) = const * N(mx,matSxx), s.t. A*x-b<0, Aeq*x-beq=0
    %
    % Usage
    %   x = randcg(mx, matSxx, A, b, Aeq, beq) draws a random vector from the
    %   constrained multivariate Gaussian density.
    %
    % Input
    %   mx              Mean vector
    %   matSxx          Covariance matrix
    %   matA, vec_b     Matrix and vector specifying inequality constraints
    %   matAeq, vec_beq Matrix and vector specifying equality constraints

    % Copyright 2009-2010 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
    %                     Morten Arngren, ma@imm.dtu.dk, phd.arngren.com

    % Dimensionality of data
    N = size(mx, 1);

    % Orthogonal basis of equality constraints + new origin
    if isempty(matAeq) % No equality constraints: use standard basis
        matP  = zeros(0, N);
        matK  = eye(N);
        cx = zeros(N, 1);
    else
        [matU, matS] = svd(matAeq');
        [m, n] = size(matAeq);
        if m > 1
            s = diag(matS); 
        elseif m == 1
            s = matS(1);
        else
            s = 0; 
        end
        tol   = max(m, n) * max(s) * eps;
        r     = sum(s>tol);
        matP  = matU(:, 1:r)';
        matK  = matU(:, r+1:end)';
        cx    = matAeq \ vec_beq;
    end
    if isempty(matA) % No inequality constraints
        matA = zeros(0,N);
        vec_b = zeros(0,1);
    end

    % Dimensionality of space that satisfies eq. constraints
    M  = N - size(matP, 1);

    W  = matK * (eye(N) - matSxx * matP' * ((matP * matSxx * matP') \ matP));
    my = W * bsxfun(@minus, mx, cx);
    matL  = chol(W * matSxx * matK');

    % Start point
    mat_w = matL' \ bsxfun(@minus, matK * bsxfun(@minus, x0, cx), my);

    % Precomputations for bounds
    matE = matA * matK' * matL';
    e = bsxfun(@minus, vec_b, matA * bsxfun(@plus, matK' * my, cx));

    % Loop over Gibbs sweeps
    for t = 1:T
        % Loop over individual elements
        for m = 1:M
            % All indices except m
            nm = [1:m-1 m+1:M];

            % Compute lower and upper bound
            mat_n  = e - matE(:, nm) * mat_w(nm, :); %-EW
            vec_d  = matE(:, m);
            lb = max(bsxfun(@rdivide, mat_n(vec_d>0,:), vec_d(vec_d>0,:)), [], 1);
            if isempty(lb)
                lb = -inf; 
            end
            ub = min(bsxfun(@rdivide, mat_n(vec_d<0,:), vec_d(vec_d<0,:)), [], 1);
            if isempty(ub)
                ub = inf; 
            end

            % Draw from truncated Gaussian density
            mat_w(m, :) = randstgs(lb, ub, mat_w(m,:));
        end
    end

    % Final result mapped back to the original space
    x = bsxfun(@plus, matK' * bsxfun(@plus, matL' * mat_w, my), cx);
end


function vec_x = randstgs(l, u, vec_x)
    % RANDTG Random numbers from standard truncated Gaussian density
    %   p(x) = const * exp(-x^2/2) *I(l < x < u) using slice sampling
    %
    % Usage
    %   x = randtg(l, u, x0) returns an array of random numbers chosen
    %     from the truncated Gaussian density. The size of x is the maximum
    %     common size of the parameters.
    %
    % Input
    %   l, u         Truncation, l < x < u

    % Copyright 2009-2010 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
    %                     Morten Arngren, ma@imm.dtu.dk, phd.arngren.com

    sz = size(l);
    z  = bsxfun(@plus, -0.5 * vec_x .^ 2, log(rand(sz)));
    s  = sqrt(-2 * z);
    ll = bsxfun(@max, -s, l);
    uu = bsxfun(@min, s, u);
    vec_x  = rand(sz) .* (uu-ll) + ll;
end