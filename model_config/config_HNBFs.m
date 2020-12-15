function [MaxItr, prior, stop_criteria] = config_HNBFs(DATA, MODEL)
    switch DATA
        case 'SmallToy'
            if strcmp(MODEL, 'HNBF')
                % HNBF ---------------------------------
                MaxItr = 100;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF ---------------------------
                MaxItr = 100;
                stop_criteria = 1;
            else
                % FastHNBF -----------------------------
                MaxItr = 100;
                stop_criteria = 1;
            end
            prior_inf = [0.3, 0.3, 0.3];
            prior_dis = [1e3, 1e8, 100, 50];
        case 'SmallToyML'
            if strcmp(MODEL, 'HNBF')
                % HNBF ---------------------------------
                MaxItr = 100;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 100;
                stop_criteria = 1;
            else
                % FastHNBF ----------------------------
                MaxItr = 100;
                stop_criteria = 1;
            end
            prior_inf = [0.3, 0.3, 0.3];
            prior_dis = [1e3, 1e8, 100, 50];
        case 'ML50'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 100;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 100;
                stop_criteria = 1;
            else
                % FastHNBF ----------------------------
                MaxItr = 100;
                stop_criteria = 1;
            end
            prior_inf = [0.3, 0.3, 0.3];
            prior_dis = [1e3, 1e8, 100, 50];
        case 'MovieLens100K'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            else
                % FastHNBF ----------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            end
            prior_inf = [0.3, 0.1, 1];
            prior_dis = [1e1, 1e8, 1, 1];
        case 'MovieLens1M'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 400;
                stop_criteria = 1;
            else
                % FastHNBF ----------------------------
                MaxItr = 400;
                stop_criteria = 1e-2;
            end
            prior_inf = [0.3, 0.1, 1];
            prior_dis = [1e1, 1e8, 1, 1];
        case 'MovieLens20M'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 600;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 600;
                stop_criteria = 1;
            else
                % FastHNBF ----------------------------
                MaxItr = 600;
                stop_criteria = 1;
            end
            prior_inf = [0.3, 0.1, 1];
            prior_dis = [1e2, 1e12, 0.5, 1];
        case 'Jester2'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            else
                % FastHNBF ----------------------------
                MaxItr = 600;
                stop_criteria = 1e-8;
            end
            prior_inf = [0.3, 0.01, 0.01];
            prior_dis = [1e1, 1e7, 1, 1];
        case 'ModCloth'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 400;
                stop_criteria = 1e-5;
            else
                % FastHNBF ----------------------------
                MaxItr = 20;
                check_step = 1;
                stop_criteria = 1e-8;
            end
            prior_inf = [0.3, 0.1, 1];
            prior_dis = [1e1, 1e7, 1, 1];
        case 'EachMovie'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 400;
                stop_criteria = 1;
            else
                % FastHNBF ----------------------------
                MaxItr = 400;
                stop_criteria = 1e-2;
            end
            prior_inf = [0.3, 0.1, 0.1];
            prior_dis = [1e1, 1e7, 1, 1];
        case 'LastFm2K'
            % mean(vs_X_train) = 746.2543
            % std(vs_X_train) = 3758.5
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 500;
                stop_criteria = 0.1;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 1600;
                stop_criteria = 0.1;
            else
                % FastHNBF ----------------------------
                MaxItr = 600;
                stop_criteria = 1e-3;
            end
            prior_inf = [3, 0.1, 0.1];
            prior_dis = [1e1, 1e8, 100, 100];
        case 'LastFm1K'
            % mean(vs_X_train) = 21.3046
            % std(vs_X_train) = 118.4657
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1e-3;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 600;
                stop_criteria = 1e-3;
            else
                % FastHNBF ----------------------------
                MaxItr = 600;
                stop_criteria = 1e-3;
            end
            prior_inf = [3, 0.1, 0.1];
            prior_dis = [1e1, 1e8, 20, 20];
        case 'LastFm360K'
            % mean(vs_X_train) = 215.2265
            % std(vs_X_train) = 611.0276
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
                MaxItr = 400;
                stop_criteria = 1e-3;
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
                MaxItr = 600;
                stop_criteria = 1e-3;
            else
                % FastHNBF ----------------------------
                MaxItr = 600;
                stop_criteria = 1e-3;
            end
            prior_inf = [3, 1, 0.1];
            prior_dis = [1e2, 1e12, 50, 50];
        case 'LastFm360K_2K'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
            else
                % FastHNBF ----------------------------
            end
        case 'EchoNest'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
            else
                % FastHNBF ----------------------------
            end
        case 'ML100KPos'
            if strcmp(MODEL, 'HNBF')
                % HNBF --------------------------------
            elseif strcmp(MODEL, 'FactorHNBF')
                % FactorHNBF --------------------------
            else
                % FastHNBF ----------------------------
            end
    end
    
    if strcmp(MODEL, 'HNBF')
        % HNBF ---------------------------------
        prior = [prior_inf prior_inf prior_dis(3:4)];
    elseif strcmp(MODEL, 'FactorHNBF')
        % FactorHNBF ---------------------------
        prior = [prior_inf prior_inf prior_dis(1:2) prior_dis(1:2)];
    else
        % FastHNBF -----------------------------
        prior = [prior_inf prior_inf prior_dis(1:2) prior_dis];
    end
    
end