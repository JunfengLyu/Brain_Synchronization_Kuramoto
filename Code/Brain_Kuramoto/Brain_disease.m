clear; clc; close all;

baseDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(baseDir, 'data');
figDir = fullfile(baseDir, '..', '..', 'Report', 'Figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end
rng(42);

groups = {'CN','EMCI','LMCI','AD'};
cols = twilight_colors(numel(groups));
[meansFull, counts] = load_group_means(dataDir, groups, [], false);

make_fig11(meansFull, counts, groups, figDir);
make_fig12(meansFull, groups, cols, figDir);
make_fig13(meansFull, counts, groups, cols, figDir);
make_fig14(dataDir, groups, cols, figDir);

fprintf('Brain_disease.m finished: Fig.11-14 are generated from real cohort connectivity data.\n');

function make_fig11(means, counts, groups, figDir)
fig = figure('Color', 'w', 'Position', [100 100 820 225]);
tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for g = 1:numel(groups)
    ax = nexttile;
    imagesc(ax, means.(groups{g}), [0 1]); axis(ax, 'image'); axis(ax, 'off');
    if exist('turbo', 'file'), colormap(ax, turbo(256)); else, colormap(ax, parula(256)); end
    text(ax, -0.18, 1.05, char('A'+g-1), 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    text(ax, 0.04, 0.95, sprintf('%s (n=%d)', groups{g}, counts.(groups{g})), ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 8.5, 'Color', 'w', 'VerticalAlignment', 'top');
end
cb = colorbar(ax, 'Location', 'eastoutside'); cb.Label.String = 'Normalized structural connectivity';
exportgraphics(fig, fullfile(figDir, '11_ad_connectome_evolution.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig12(means, groups, cols, figDir)
CN = means.CN;
diseaseGroups = groups(2:end);
globalMax = 0;
rawLesions = struct();
for i = 1:numel(diseaseGroups)
    g = diseaseGroups{i};
    rawLesions.(g) = sum(max(CN - means.(g), 0), 2);
    globalMax = max(globalMax, max(rawLesions.(g)));
end
metrics = graph_metrics(CN);
metricNames = fieldnames(metrics);
fig = figure('Color', 'w', 'Position', [100 100 740 520]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for m = 1:4
    ax = nexttile; hold on;
    x = metrics.(metricNames{m});
    leg = cell(1, numel(diseaseGroups));
    for i = 1:numel(diseaseGroups)
        g = diseaseGroups{i};
        y = rawLesions.(g) / max(globalMax, eps);
        scatter(x, y, 14, cols(i+1,:), 'filled', 'MarkerFaceAlpha', 0.45);
        p = polyfit(x, y, 1);
        xx = linspace(min(x), max(x), 80);
        plot(xx, polyval(p, xx), 'LineWidth', 1.5, 'Color', cols(i+1,:));
        r = corr_local(x(:), y(:));
        r2 = r^2;
        F = (r2 * (numel(x)-2)) / max(1-r2, eps);
        pv = f_upper_pvalue(F, 1, numel(x)-2);
        leg{i} = sprintf('%s: F=%.1f, p=%s', g, F, p_string(pv));
    end
    text(-0.13, 1.03, char('A'+m-1), 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    xlabel(['Baseline ' pretty_metric(metricNames{m})]);
    ylabel('Lesion Count (Normalised)');
    legend(leg, 'Location', 'northwest', 'Box', 'off', 'FontSize', 6.7);
    style_axes(ax);
end
exportgraphics(fig, fullfile(figDir, '12_topological_lesion_regression.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig13(means, counts, groups, cols, figDir)
lambdas = linspace(0, 0.05, 151);
R = zeros(numel(groups), numel(lambdas));
for g = 1:numel(groups)
    fprintf('Fig13: %s\n', groups{g});
    for k = 1:numel(lambdas)
        R(g,k) = kuramoto_order(means.(groups{g}), lambdas(k), 2.5, 0.2, 80, 4000, 2000, 0.01, 42);
    end
end
fig = figure('Color', 'w', 'Position', [100 100 545 310]); hold on;
patch([0.015 0.030 0.030 0.015], [0 0 1.05 1.05], [0.16 0.65 0.72], ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', 'DisplayName', 'Critical Regime (0.01-0.03)');
for g = 1:numel(groups)
    plot(lambdas, R(g,:), 'LineWidth', 1.5, 'Color', cols(g,:), ...
        'DisplayName', sprintf('%s (n=%d)', groups{g}, counts.(groups{g})));
end
xlabel('Cortical Coupling Factor, $\lambda$', 'Interpreter', 'latex');
ylabel('Order Parameter $R$', 'Interpreter', 'latex');
xlim([0 0.05]); ylim([0 1.05]); grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.30);
legend('Location', 'southeast', 'Box', 'off', 'FontSize', 7.2);
style_axes(gca);
exportgraphics(fig, fullfile(figDir, '13_ad_phase_transition_delay.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig14(dataDir, groups, cols, figDir)
lambdas = 0:0.01:0.6;
[means84, ~] = load_group_means(dataDir, groups, 84, true);
Mcn = means84.CN; Mad = means84.AD; N = size(Mcn,1);
[V, D] = eig(Mcn); [~, idxMax] = max(diag(D));
eigCent = abs(V(:,idxMax));
[~, hubIdx] = sort(eigCent, 'descend');
numHubs = max(1, round(0.10*N));
topHubs = hubIdx(1:numHubs);
rng(42); randomNodes = randperm(N, numHubs);
Mhub = Mad; Mhub(topHubs,:) = Mcn(topHubs,:); Mhub(:,topHubs) = Mcn(:,topHubs);
Mrand = Mad; Mrand(randomNodes,:) = Mcn(randomNodes,:); Mrand(:,randomNodes) = Mcn(:,randomNodes);
rng(100); omega0 = 0.25 * randn(N,1); theta0 = rand(N,1)*2*pi;
curves = zeros(4, numel(lambdas));
mats = {Mcn, Mhub, Mrand, Mad};
for i = 1:4
    fprintf('Fig14 curve %d\n', i);
    for k = 1:numel(lambdas)
        curves(i,k) = kuramoto_order_fixed(mats{i}, lambdas(k), omega0, theta0, 0.25, 1, 3000, 1800, 0.01, 42);
    end
end
fig = figure('Color', 'w', 'Position', [100 100 545 325]); hold on;
plot(lambdas, curves(1,:), '-^', 'Color', cols(1,:), 'LineWidth', 1.5, 'MarkerSize', 3.2, 'MarkerFaceColor', cols(1,:), 'DisplayName', 'Healthy Baseline (CN)');
plot(lambdas, curves(2,:), '-o', 'Color', cols(3,:), 'LineWidth', 1.5, 'MarkerSize', 3.4, 'MarkerFaceColor', cols(3,:), 'DisplayName', 'Targeted Rescue (Top 10 Hubs)');
plot(lambdas, curves(3,:), '--s', 'Color', cols(2,:), 'LineWidth', 1.5, 'MarkerSize', 3.2, 'MarkerFaceColor', cols(2,:), 'DisplayName', 'Random Rescue (10 Non-Hubs)');
plot(lambdas, curves(4,:), ':', 'Color', cols(4,:), 'LineWidth', 1.5, 'DisplayName', 'Pathological Baseline (AD)');
xlabel('Coupling Strength, $\lambda$', 'Interpreter', 'latex');
ylabel('Global Synchronization, $R$', 'Interpreter', 'latex');
xlim([0 0.6]); ylim([0 1.05]); grid on; set(gca, 'GridAlpha', 0.15);
legend('Location', 'southeast', 'Box', 'off', 'FontSize', 7);
style_axes(gca);
exportgraphics(fig, fullfile(figDir, '14_perturbation_rescue_experiment.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function [means, counts] = load_group_means(dataDir, groups, nNodes, tailMode)
means = struct(); counts = struct();
for g = 1:numel(groups)
    files = dir(fullfile(dataDir, groups{g}, '*.csv'));
    sumMat = []; valid = 0;
    for i = 1:numel(files)
        path = fullfile(files(i).folder, files(i).name);
        M = read_csv_matrix(path);
        if isempty(M) || size(M,1) ~= size(M,2), continue; end
        if ~isempty(nNodes)
            if size(M,1) < nNodes, continue; end
            if tailMode
                M = M(end-nNodes+1:end, end-nNodes+1:end);
            else
                M = M(1:nNodes, 1:nNodes);
            end
        end
        M = log1p(max(M, 0));
        M = (M + M') / 2; M(1:size(M,1)+1:end) = 0;
        if isempty(sumMat)
            sumMat = zeros(size(M));
        end
        if isequal(size(M), size(sumMat))
            sumMat = sumMat + M; valid = valid + 1;
        end
    end
    if valid == 0, error('No valid CSV matrices for %s', groups{g}); end
    A = sumMat / valid;
    if max(A(:)) > 0, A = A / max(A(:)); end
    A(1:size(A,1)+1:end) = 0;
    means.(groups{g}) = A;
    counts.(groups{g}) = valid;
end
end

function M = read_csv_matrix(path)
try
    M = readmatrix(path);
catch
    try
        M = csvread(path);
    catch
        M = [];
    end
end
M = double(M);
M(~isfinite(M)) = 0;
end

function metrics = graph_metrics(W)
N = size(W,1);
[V,D] = eig(W); [~, imax] = max(diag(D));
metrics.Eigenvector_Centrality = abs(V(:,imax)) / max(abs(V(:,imax)) + eps);
Wc = W.^(1/3);
clust = zeros(N,1);
for i = 1:N
    k = sum(W(i,:) > 0);
    if k > 1
        clust(i) = (Wc(i,:) * Wc * Wc(:,i)) / (k*(k-1));
    end
end
metrics.Clustering_Coefficient = clust;
thr = percentile_vector(W(:), 70);
A = double(W > thr); A(1:N+1:end) = 0;
eff = zeros(N,1);
for i = 1:N
    nb = find(A(i,:));
    if numel(nb) > 1
        Dsub = shortest_unweighted(A(nb,nb));
        invD = 1 ./ Dsub; invD(~isfinite(invD)) = 0; invD(1:size(invD,1)+1:end) = 0;
        eff(i) = sum(invD(:)) / (numel(nb)*(numel(nb)-1));
    end
end
metrics.Local_Efficiency = eff;
L = diag(sum(A,2)) - A;
[Ve,De] = eig(L); [~, ord] = sort(diag(De));
v2 = Ve(:,ord(min(2,N))); v3 = Ve(:,ord(min(3,N)));
comm = ones(N,1);
comm(v2 > 0 & v3 <= 0) = 2; comm(v2 <= 0 & v3 > 0) = 3; comm(v2 <= 0 & v3 <= 0) = 4;
part = zeros(N,1);
for i = 1:N
    ki = sum(W(i,:));
    if ki > 0
        s = 0;
        for c = 1:4
            s = s + (sum(W(i,comm==c))/ki)^2;
        end
        part(i) = 1 - s;
    end
end
metrics.Participation_Coefficient = part;
end

function D = shortest_unweighted(A)
n = size(A,1);
D = inf(n); D(1:n+1:end) = 0; D(A > 0) = 1;
for k = 1:n
    D = min(D, D(:,k) + D(k,:));
end
end

function R = kuramoto_order(M, lambda, omegaStd, noise, scale, steps, burn, dt, seed)
rng(seed);
N = size(M,1);
omega = omegaStd * randn(N,1);
theta = rand(N,1) * 2*pi;
R = kuramoto_order_fixed(M, lambda, omega, theta, noise, scale, steps, burn, dt, seed);
end

function R = kuramoto_order_fixed(M, lambda, omega, theta, noise, scale, steps, burn, dt, seed)
rng(seed);
Rvals = zeros(steps-burn+1,1); idx = 1;
for s = 1:steps
    phase = theta' - theta;
    drift = omega + scale * lambda * sum(M .* sin(phase), 2);
    theta = theta + drift * dt + noise * sqrt(dt) * randn(size(theta));
    if s >= burn
        Rvals(idx) = abs(mean(exp(1i*theta)));
        idx = idx + 1;
    end
end
R = mean(Rvals(1:idx-1));
end

function style_axes(ax)
set(ax, 'FontName', 'Arial', 'FontSize', 8.5, 'LineWidth', 1.1, 'TickDir', 'out', 'Box', 'off');
end

function cols = twilight_colors(n)
base = [0.8858 0.8500 0.8879; 0.6443 0.4390 0.7389; 0.1849 0.0794 0.2131; ...
        0.2453 0.2878 0.5373; 0.2890 0.5650 0.7595; 0.7590 0.7060 0.5680; ...
        0.8858 0.8500 0.8879];
x = linspace(0,1,size(base,1)); xi = linspace(0.08,0.88,n);
cols = interp1(x, base, xi);
end

function q = percentile_vector(v, pct)
v = sort(v(:));
v = v(isfinite(v));
if isempty(v), q = 0; return; end
pos = 1 + (numel(v)-1) * pct/100;
lo = floor(pos); hi = ceil(pos);
if lo == hi
    q = v(lo);
else
    q = v(lo) + (pos-lo) * (v(hi)-v(lo));
end
end

function r = corr_local(x, y)
mask = isfinite(x) & isfinite(y);
x = x(mask); y = y(mask);
if numel(x) < 2 || std(x) == 0 || std(y) == 0
    r = 0;
else
    C = corrcoef(x, y);
    r = C(1,2);
end
end

function s = pretty_metric(name)
s = strrep(name, '_', ' ');
end

function pv = f_upper_pvalue(F, df1, df2)
x = (df1 * F) / (df1 * F + df2);
pv = 1 - betainc(x, df1/2, df2/2);
pv = max(0, min(1, pv));
end

function s = p_string(pv)
if pv < 0.001
    s = sprintf('%.2e', pv);
else
    s = sprintf('%.3f', pv);
end
end
