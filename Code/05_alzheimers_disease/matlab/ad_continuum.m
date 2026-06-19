clear; clc; close all;
addpath('../../common/matlab');
figDir = fullfile('..','..','..','Report','Figs');
dataDir = fullfile('..','..','..','data_for_section3&4');
groups = {'CN','EMCI','LMCI','AD'};
twilightCols = twilight_colors(4);
cols = {twilightCols(1,:), twilightCols(2,:), twilightCols(3,:), twilightCols(4,:)};
sigmoid = @(x,c,w,lo,hi) lo + (hi-lo) ./ (1 + exp(-(x-c)./w));

data = cell(size(groups));
counts = zeros(size(groups));
for g = 1:numel(groups)
    files = dir(fullfile(dataDir, groups{g}, '*.csv'));
    mats = {};
    minN = inf;
    for i = 1:numel(files)
        try
            M = readmatrix(fullfile(files(i).folder, files(i).name));
            if size(M,1) == size(M,2) && size(M,1) >= 60
                M(isnan(M)) = 0;
                M = log1p(M);
                M = (M + M') / 2;
                mats{end+1} = M; %#ok<SAGROW>
                minN = min(minN, size(M,1));
            end
        catch
        end
    end
    counts(g) = numel(mats);
    stack = zeros(minN, minN, counts(g));
    for i = 1:counts(g)
        stack(:,:,i) = mats{i}(1:minN, 1:minN);
    end
    data{g} = stack;
end

means = cell(size(groups));
for g = 1:numel(groups)
    means{g} = mean(data{g}, 3);
    if max(means{g}(:)) > 0, means{g} = means{g} / max(means{g}(:)); end
    means{g} = means{g} - diag(diag(means{g}));
end

%% Figure 11
fig = figure('Position', [100 100 820 225], 'Color', 'w');
for g = 1:numel(groups)
    subplot(1,4,g);
    imagesc(means{g}, [0 1]); axis image off; colormap(gca, turbo);
    title(sprintf('%s (n=%d)', groups{g}, counts(g)), 'FontSize', 9);
    text(-0.18, 1.05, char('A'+g-1), 'Units', 'normalized', 'FontSize', 11);
end
cb = colorbar('Position', [0.93 0.22 0.015 0.58]);
cb.Label.String = 'Normalized structural connectivity';
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '11_ad_connectome_evolution.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Metrics for Figure 12
CN = means{1};
n = size(CN,1);
rawDiffs = [];
lesionRaw = cell(1,3);
for g = 2:4
    lc = sum(CN - means{g}, 2);
    lesionRaw{g-1} = lc;
    rawDiffs = [rawDiffs; lc]; %#ok<AGROW>
end
globalMaxLesion = max(rawDiffs);
lesion = cell(1,3);
for g = 2:4
    lesion{g-1} = lesionRaw{g-1} / globalMaxLesion;
end

[V, ~] = eigs(CN, 1, 'largestabs');
eigCent = abs(V);
Wc = CN.^(1/3);
clust = zeros(n,1);
for i = 1:n
    k_i = sum(CN(i,:) > 0);
    if k_i > 1
        triangles = Wc(i,:) * Wc * Wc(:,i);
        clust(i) = triangles / (k_i * (k_i - 1));
    end
end
threshold = prctile(CN(:), 70);
Gbin = double(CN > threshold);
localEff = zeros(n,1);
for i = 1:n
    neighbors = find(Gbin(i,:));
    k_i = length(neighbors);
    if k_i > 1
        subA = Gbin(neighbors, neighbors);
        D = distances(graph(subA));
        invD = 1 ./ D;
        invD(isinf(invD) | isnan(invD)) = 0;
        invD(1:size(invD,1)+1:end) = 0;
        localEff(i) = sum(invD(:)) / (k_i * (k_i - 1));
    end
end
Dmat = diag(sum(Gbin, 2));
L = Dmat - Gbin;
[Veig, Dval] = eig(L);
eigVals = diag(Dval);
[~, sortIdx] = sort(eigVals);
v2 = Veig(:, sortIdx(2));
v3 = Veig(:, sortIdx(3));
communities = zeros(n,1);
communities(v2 > 0 & v3 > 0) = 1;
communities(v2 > 0 & v3 <= 0) = 2;
communities(v2 <= 0 & v3 > 0) = 3;
communities(v2 <= 0 & v3 <= 0) = 4;
part = zeros(n,1);
for i = 1:n
    k_i = sum(CN(i,:));
    if k_i > 0
        s = 0;
        for c = 1:4
            comNodes = find(communities == c);
            s = s + (sum(CN(i, comNodes)) / k_i)^2;
        end
        part(i) = 1 - s;
    end
end
metrics = {eigCent, clust, localEff, part};
metricNames = {'Eigenvector Centrality','Clustering Coefficient','Local Efficiency','Participation Coefficient'};

fig = figure('Position', [100 100 740 520], 'Color', 'w');
for m = 1:4
    subplot(2,2,m); hold on;
    x = metrics{m};
    leg = cell(1,3);
    for g = 2:4
        y = lesion{g-1};
        scatter(x, y, 18, cols{g}, 'filled', 'MarkerFaceAlpha', 0.45);
        p = polyfit(x, y, 1);
        xx = linspace(min(x), max(x), 80);
        plot(xx, polyval(p, xx), 'Color', cols{g}, 'LineWidth', 1.7);
        R = corrcoef(x, y);
        r2 = R(1,2)^2;
        df2 = length(x) - 2;
        if r2 < 1
            Fstat = (r2 * df2) / (1 - r2);
        else
            Fstat = inf;
        end
        pval = fcdf(Fstat, 1, df2, 'upper');
        if pval < 0.001
            pstr = sprintf('%.2e', pval);
        else
            pstr = sprintf('%.3f', pval);
        end
        leg{g-1} = sprintf('%s: F=%.1f, p=%s', groups{g}, Fstat, pstr);
    end
    text(-0.13, 1.03, char('A'+m-1), 'Units', 'normalized', 'FontSize', 11);
    xlabel(['Baseline ' metricNames{m}]); ylabel('Lesion Count (Normalised)');
    legend(leg, 'Location', 'northwest', 'FontSize', 6.7);
end
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '12_topological_lesion_regression.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 13
lambda = linspace(0, 0.05, 151);
R13 = zeros(numel(groups), numel(lambda));
for k = 1:numel(lambda)
    for g = 1:numel(groups)
        R13(g,k) = eval_kuramoto_original(means{g}, lambda(k), 2.5, 0.2);
    end
end
fig = figure('Position', [100 100 545 310], 'Color', 'w');
hold on;
patch([0.015 0.030 0.030 0.015], [0 0 1.05 1.05], [0.16 0.65 0.72], 'FaceAlpha', 0.10, 'EdgeColor', 'none', 'DisplayName', 'Critical Regime (0.01-0.03)');
for g = 1:4
    plot(lambda, R13(g,:), 'Color', cols{g}, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (n=%d)', groups{g}, counts(g)));
end
xlabel('Cortical Coupling Factor, \lambda'); ylabel('Order Parameter r');
xlim([0 0.05]); ylim([0 1.05]);
legend('Location', 'southeast');
grid on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.5);
exportgraphics(fig, fullfile(figDir, '13_ad_phase_transition_delay.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 14
lambda14 = 0:0.01:0.6;
Mcn84 = load_mat_secure_local(fullfile(dataDir, 'CN'), 84);
Mad84 = load_mat_secure_local(fullfile(dataDir, 'AD'), 84);
[Vhub, ~] = eigs(Mcn84, 1, 'largestabs');
metricEigen = abs(Vhub);
[~, hubIdx] = sort(metricEigen, 'descend');
numHubs = round(0.1 * 84);
topHubs = hubIdx(1:numHubs);
Mhub = Mad84;
Mhub(topHubs,:) = Mcn84(topHubs,:);
Mhub(:,topHubs) = Mcn84(:,topHubs);
rng(42);
randNodes = randperm(84, numHubs);
Mrand = Mad84;
Mrand(randNodes,:) = Mcn84(randNodes,:);
Mrand(:,randNodes) = Mcn84(:,randNodes);
rng(100);
omegaInit = randn(84,1) * 0.25;
thetaInit = rand(84,1) * 2*pi;
Rcn = zeros(size(lambda14)); Rad = Rcn; Rhub = Rcn; Rrand = Rcn;
for k = 1:numel(lambda14)
    Rcn(k) = kuramoto_rescue_original(Mcn84, lambda14(k), omegaInit, thetaInit);
    Rad(k) = kuramoto_rescue_original(Mad84, lambda14(k), omegaInit, thetaInit);
    Rhub(k) = kuramoto_rescue_original(Mhub, lambda14(k), omegaInit, thetaInit);
    Rrand(k) = kuramoto_rescue_original(Mrand, lambda14(k), omegaInit, thetaInit);
end
fig = figure('Position', [100 100 545 325], 'Color', 'w');
hold on;
plot(lambda14, Rcn, '-^', 'Color', cols{1}, 'LineWidth', 1.5, 'MarkerSize', 3.2, 'MarkerFaceColor', cols{1}, 'DisplayName', 'Healthy Baseline (CN)');
plot(lambda14, Rhub, '-o', 'Color', cols{3}, 'LineWidth', 1.5, 'MarkerSize', 3.6, 'MarkerFaceColor', cols{3}, 'DisplayName', 'Targeted Rescue (Top 10 Hubs)');
plot(lambda14, Rrand, '--s', 'Color', cols{2}, 'LineWidth', 1.5, 'MarkerSize', 3.2, 'MarkerFaceColor', cols{2}, 'DisplayName', 'Random Rescue (10 Non-Hubs)');
plot(lambda14, Rad, ':', 'Color', cols{4}, 'LineWidth', 1.5, 'DisplayName', 'Pathological Baseline (AD)');
xlabel('Coupling Strength, \lambda'); ylabel('Global Synchronization, R');
xlim([0 0.6]); ylim([0 1.05]);
legend('Location', 'southeast', 'Box', 'off');
box off; grid on; set(gca, 'LineWidth', 1.5, 'TickDir', 'out', 'GridAlpha', 0.15);
exportgraphics(fig, fullfile(figDir, '14_perturbation_rescue_experiment.png'), 'Resolution', 300, 'BackgroundColor', 'w');

function rGlobal = eval_kuramoto_original(M, lam, omegaStd, noiseStrength)
rng(42);
N = size(M, 1);
omega = randn(N, 1) * omegaStd;
theta = rand(N, 1) * 2*pi;
tMax = 40;
dt = 0.01;
steps = floor(tMax / dt);
steadySteps = floor(steps / 2);
thetaSteady = zeros(N, steadySteps);
noiseFactor = noiseStrength * sqrt(dt);
for step = 1:steps
    phaseDiff = theta' - theta;
    dtheta = omega + 80 * lam * sum(M .* sin(phaseDiff), 2);
    theta = theta + dtheta * dt + noiseFactor * randn(N, 1);
    if step > steadySteps
        thetaSteady(:, step - steadySteps) = theta;
    end
end
rGlobal = mean(abs(mean(exp(1i * thetaSteady), 1)));
end

function R = kuramoto_rescue_original(M, lambdaVal, omegaInit, thetaInit)
rng(42);
noiseStrength = 0.25;
dt = 0.01;
tMax = 30;
steps = floor(tMax / dt);
steadyStart = floor(steps * 0.6);
omega = omegaInit;
theta = thetaInit;
noiseFactor = noiseStrength * sqrt(dt);
Rvec = zeros(1, steps - steadyStart + 1);
idx = 1;
for step = 1:steps
    phaseDiff = theta' - theta;
    drift = omega + lambdaVal * sum(M .* sin(phaseDiff), 2);
    theta = theta + drift * dt + noiseFactor * randn(size(theta));
    if step >= steadyStart
        Rvec(idx) = abs(mean(exp(1i * theta)));
        idx = idx + 1;
    end
end
R = mean(Rvec);
end

function meanMat = load_mat_secure_local(folder, N)
files = dir(fullfile(folder, '*.csv'));
sumMat = zeros(N, N);
validCount = 0;
for j = 1:length(files)
    try
        temp = readmatrix(fullfile(folder, files(j).name));
        if isempty(temp) || size(temp, 1) < N || size(temp, 2) < N, continue; end
        mat = temp(end-N+1:end, end-N+1:end);
        mat = log1p(mat);
        mat = (mat + mat') / 2;
        sumMat = sumMat + mat;
        validCount = validCount + 1;
    catch
        continue;
    end
end
meanMat = sumMat / validCount;
if max(meanMat(:)) > 0, meanMat = meanMat / max(meanMat(:)); end
meanMat(1:N+1:end) = 0;
end

function cmap = magma_or_hot()
try
    cmap = magma(256);
catch
    cmap = hot(256);
end
end

function y = clip01(y)
y = max(0, min(1, y));
end

function cols = twilight_colors(n)
base = [ ...
    0.8858 0.8500 0.8879
    0.6443 0.4390 0.7389
    0.1849 0.0794 0.2131
    0.2453 0.2878 0.5373
    0.2890 0.5650 0.7595
    0.7590 0.7060 0.5680
    0.8858 0.8500 0.8879];
x = linspace(0, 1, size(base, 1));
xi = linspace(0.08, 0.88, n);
cols = interp1(x, base, xi);
end
