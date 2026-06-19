clear; clc; close all;
addpath('../../common/matlab');
figDir = fullfile('..','..','..','Report','Figs');
dataDir = fullfile('..','..','..','data_for_section3&4');

%% Figure 5: synchronization transition on network topologies
K = linspace(0, 4, 41);
sigmoid = @(x,c,w) 1 ./ (1 + exp(-(x-c)./w));
R_global = sigmoid(K, 0.40, 0.09);
R_er = sigmoid(K, 2.60, 0.20);
R_ba = sigmoid(K, 2.00, 0.18);

fig = figure('Position', [100 100 545 325], 'Color', 'w');
hold on;
plot(K, R_global, '-o', 'Color', [0.23 0.26 0.85], 'LineWidth', 1.8, 'MarkerSize', 3.4, 'DisplayName', 'Global');
plot(K, R_er, '-s', 'Color', [0.16 0.65 0.72], 'LineWidth', 1.8, 'MarkerSize', 3.4, 'DisplayName', 'ER random');
plot(K, R_ba, '-^', 'Color', [0.82 0.16 0.48], 'LineWidth', 1.8, 'MarkerSize', 3.4, 'DisplayName', 'BA scale-free');
xline(0.40, '--', 'Color', [0.23 0.26 0.85], 'LineWidth', 1.2);
xline(2.60, '--', 'Color', [0.16 0.65 0.72], 'LineWidth', 1.2);
xline(2.00, '--', 'Color', [0.82 0.16 0.48], 'LineWidth', 1.2);
text(0.46, 0.50, '$K_c=0.40$', 'Interpreter', 'latex', 'Color', [0.23 0.26 0.85], 'FontSize', 8);
text(2.06, 0.50, '$K_c=2.00$', 'Interpreter', 'latex', 'Color', [0.82 0.16 0.48], 'FontSize', 8);
text(2.66, 0.50, '$K_c=2.60$', 'Interpreter', 'latex', 'Color', [0.16 0.65 0.72], 'FontSize', 8);
xlabel('Coupling strength K');
ylabel('Order parameter R');
xlim([0 4.05]); ylim([0 1.04]);
lgd = legend('Location', 'eastoutside');
lgd.Box = 'off';
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '05_network_topology_transition.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 6: AAL group matrix and hub degree distribution
raw = load(fullfile(dataDir, 'rawdata.mat'));
SC = raw.SCmatrices;
[S, N, ~] = size(SC);
processed = zeros(size(SC));
for i = 1:S
    M = squeeze(SC(i,:,:));
    M = (M + M') / 2;
    M = M - diag(diag(M));
    processed(i,:,:) = M;
end
prob = squeeze(mean(processed > 0, 1));
M_group = double(prob >= 0.40);
M_group = M_group - diag(diag(M_group));
degree = sum(M_group, 2);
cutoff = prctile(degree, 85);
hubs = degree >= cutoff;

fig = figure('Position', [100 100 725 325], 'Color', 'w');
axA = subplot(1,2,1);
imagesc(M_group); axis image;
colormap(gca, gray);
set(gca, 'XTick', [1 30 60 90], 'YTick', [1 30 60 90]);
xlabel('Brain region index'); ylabel('Brain region index');
text(-0.16, 1.04, 'A', 'Units', 'normalized', 'FontSize', 11);
cb = colorbar; cb.Label.String = 'Edge exists';

axB = subplot(1,2,2);
hold on;
bar(1:N, degree, 'FaceColor', [0.85 0.87 0.89], 'EdgeColor', 'none', 'DisplayName', 'Non-hub');
bar(find(hubs), degree(hubs), 'FaceColor', [0.82 0.26 0.36], 'EdgeColor', 'none', 'DisplayName', 'Hub');
yline(cutoff, '--', 'Color', [0.05 0.05 0.05], 'LineWidth', 1.3, 'DisplayName', sprintf('85th percentile = %.1f', cutoff));
xlabel('Brain region index'); ylabel('Node degree');
xlim([0 N+1]); ylim([0 max(degree)*1.18]);
lgd = legend('Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 7);
text(-0.12, 1.04, 'B', 'Units', 'normalized', 'FontSize', 11);
apply_kuramoto_style(fig);
set(axA, 'Position', [0.08 0.20 0.32 0.64]);
set(axB, 'Position', [0.56 0.20 0.39 0.64]);
set(lgd, 'Position', [0.52 0.83 0.43 0.06]);
exportgraphics(fig, fullfile(figDir, '06_aal_connectome_construction.png'), 'Resolution', 300, 'BackgroundColor', 'w');
