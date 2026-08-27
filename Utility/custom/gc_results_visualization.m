load('gc_results_backup.mat')
% ---------------------------------------------------------------------
% Visualization:
% 1) Directionality index sensitivity (DI / WDI) with mean ± SEM
% 2) Total causal flow sensitivity (L2/3->L5 vs L5->L2/3) with mean ± SEM
%    and significance markers
% ---------------------------------------------------------------------
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 7);

color_green = [0.4660, 0.6740, 0.1880]; % L2/3 -> L5
color_blue  = [0.0000, 0.4470, 0.7410]; % L5 -> L2/3
color_gray1 = [0.35 0.35 0.35];         % DI
color_gray2 = [0.60 0.60 0.60];         % WDI

get_star = @(p) repmat('*', 1, (p<0.05) + (p<0.01) + (p<0.001));

for c = 1:length(cond_names)
    cond = cond_names{c};
    
    % -------------------------------------------------------------
    % Collect summary values across PC numbers
    % -------------------------------------------------------------
    mean_DI   = nan(1, length(num_pcs_list));
    sem_DI    = nan(1, length(num_pcs_list));
    mean_WDI  = nan(1, length(num_pcs_list));
    sem_WDI   = nan(1, length(num_pcs_list));
    n_sess    = nan(1, length(num_pcs_list));
    p_DI0     = nan(1, length(num_pcs_list));   % DI vs 0
    p_WDI0    = nan(1, length(num_pcs_list));   % WDI vs 0
    
    mean_A23  = nan(1, length(num_pcs_list));
    sem_A23   = nan(1, length(num_pcs_list));
    mean_A5   = nan(1, length(num_pcs_list));
    sem_A5    = nan(1, length(num_pcs_list));
    p_Apair   = nan(1, length(num_pcs_list));   % L2/3->L5 vs L5->L2/3
    
    for p_i = 1:length(num_pcs_list)
        pc_field = sprintf('PC%d', num_pcs_list(p_i));
        
        data_DI   = stats_GC.(pc_field).(cond).DI;
        data_WDI  = stats_GC.(pc_field).(cond).WDI;
        data_A23  = stats_GC.(pc_field).(cond).A_23_to_5;
        data_A5   = stats_GC.(pc_field).(cond).A_5_to_23;
        
        if ~isempty(data_DI)
            mean_DI(p_i) = mean(data_DI, 'omitnan');
            sem_DI(p_i)  = std(data_DI, 0, 'omitnan') / sqrt(length(data_DI));
            n_sess(p_i)  = length(data_DI);
            if length(data_DI) >= 3
                p_DI0(p_i) = signrank(data_DI);
            end
        end
        
        if ~isempty(data_WDI)
            mean_WDI(p_i) = mean(data_WDI, 'omitnan');
            sem_WDI(p_i)  = std(data_WDI, 0, 'omitnan') / sqrt(length(data_WDI));
            if length(data_WDI) >= 3
                p_WDI0(p_i) = signrank(data_WDI);
            end
        end
        
        if ~isempty(data_A23)
            mean_A23(p_i) = mean(data_A23, 'omitnan');
            sem_A23(p_i)  = std(data_A23, 0, 'omitnan') / sqrt(length(data_A23));
        end
        
        if ~isempty(data_A5)
            mean_A5(p_i)  = mean(data_A5, 'omitnan');
            sem_A5(p_i)   = std(data_A5, 0, 'omitnan') / sqrt(length(data_A5));
        end
        
        if ~isempty(data_A23) && ~isempty(data_A5) && length(data_A23) >= 3 && length(data_A5) >= 3
            p_Apair(p_i) = signrank(data_A23, data_A5);
        end
    end
    
    % =============================================================
    % Figure 1: DI / WDI sensitivity with mean ± SEM
    % =============================================================
    fig1 = figure('Units', 'centimeters', 'Position', [2, 2, 10, 6], 'Color', 'w');
    hold on;
    
    errorbar(num_pcs_list, mean_DI,  sem_DI,  '-o', ...
        'LineWidth', 1.2, 'MarkerSize', 4, ...
        'Color', color_gray1, 'MarkerFaceColor', color_gray1);
    
    errorbar(num_pcs_list, mean_WDI, sem_WDI, '-s', ...
        'LineWidth', 1.2, 'MarkerSize', 4, ...
        'Color', color_gray2, 'MarkerFaceColor', color_gray2);
    
    yline(0, '--k', 'LineWidth', 0.5);
    
    xlabel('Number of PCs per layer');
    ylabel('Directionality index');
    title(sprintf('GC Directionality Sensitivity: %s', cond));
    legend({'Unweighted DI', 'Variance-weighted DI'}, 'Location', 'best', 'Box', 'off');
    set(gca, 'LineWidth', 0.5, 'Box', 'off');
    
    % Add N labels
    y_lim = ylim;
    for p_i = 1:length(num_pcs_list)
        if ~isnan(n_sess(p_i))
            text(num_pcs_list(p_i), y_lim(1) + 0.05 * range(y_lim), ...
                sprintf('N=%d', n_sess(p_i)), ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 6);
        end
    end
    
    % Add DI-vs-0 significance marker above each point
    y_lim = ylim;
    for p_i = 1:length(num_pcs_list)
        if ~isnan(p_DI0(p_i))
            sig_str = get_star(p_DI0(p_i));
            if isempty(sig_str), sig_str = 'n.s.'; end
            text(num_pcs_list(p_i), mean_DI(p_i) + sem_DI(p_i) + 0.06 * range(y_lim), ...
                sig_str, 'HorizontalAlignment', 'center', 'FontSize', 6, 'Color', color_gray1);
        end
    end
    
    pdf_filename1 = fullfile(output_dir, sprintf('%s_GC_PC_Sensitivity_DI_%s.pdf', ...
        strjoin(Date(1)), cond));
    exportgraphics(fig1, pdf_filename1, 'ContentType', 'vector');
    close(fig1);
    
    % =============================================================
    % Figure 2: Total Causal Flow sensitivity with mean ± SEM
    % =============================================================
    fig2 = figure('Units', 'centimeters', 'Position', [2, 2, 11, 6], 'Color', 'w');
    hold on;
    
    x1 = num_pcs_list - 0.18;
    x2 = num_pcs_list + 0.18;
    
    errorbar(x1, mean_A23, sem_A23, '-o', ...
        'LineWidth', 1.2, 'MarkerSize', 4, ...
        'Color', color_green, 'MarkerFaceColor', color_green);
    
    errorbar(x2, mean_A5, sem_A5, '-o', ...
        'LineWidth', 1.2, 'MarkerSize', 4, ...
        'Color', color_blue, 'MarkerFaceColor', color_blue);
    
    xlabel('Number of PCs per layer');
    ylabel('Total Causal Flow (Sum of F)');
    title(sprintf('GC Total Causal Flow Sensitivity: %s', cond));
    legend({'L2/3 \rightarrow L5', 'L5 \rightarrow L2/3'}, 'Location', 'best', 'Box', 'off');
    set(gca, 'LineWidth', 0.5, 'Box', 'off');
    
    % Add N labels near bottom
    y_lim = ylim;
    for p_i = 1:length(num_pcs_list)
        if ~isnan(n_sess(p_i))
            text(num_pcs_list(p_i), y_lim(1) + 0.05 * range(y_lim), ...
                sprintf('N=%d', n_sess(p_i)), ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 6);
        end
    end
    
    % Add significance markers for L2/3->L5 vs L5->L2/3 at each PC number
    y_lim = ylim;
    for p_i = 1:length(num_pcs_list)
        if ~isnan(p_Apair(p_i)) && ~isnan(mean_A23(p_i)) && ~isnan(mean_A5(p_i))
            sig_str = get_star(p_Apair(p_i));
            if isempty(sig_str), sig_str = 'n.s.'; end
            
            curr_max = max([mean_A23(p_i) + sem_A23(p_i), mean_A5(p_i) + sem_A5(p_i)]);
            bar_y = curr_max + 0.06 * range(y_lim);
            text_y = curr_max + 0.10 * range(y_lim);
            
            plot([x1(p_i) x1(p_i) x2(p_i) x2(p_i)], ...
                 [bar_y-0.01*range(y_lim) bar_y bar_y bar_y-0.01*range(y_lim)], ...
                 '-k', 'LineWidth', 0.5);
            
            text(num_pcs_list(p_i), text_y, sig_str, ...
                'HorizontalAlignment', 'center', 'FontSize', 6);
        end
    end
    
    pdf_filename2 = fullfile(output_dir, sprintf('%s_GC_PC_Sensitivity_TotalFlow_%s.pdf', ...
        strjoin(Date(1)), cond));
    exportgraphics(fig2, pdf_filename2, 'ContentType', 'vector');
    close(fig2);
end

