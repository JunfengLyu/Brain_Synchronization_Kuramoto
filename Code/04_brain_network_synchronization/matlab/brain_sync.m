clear; clc; close all;
addpath('../../common/matlab');
figDir = fullfile('..','..','..','Report','Figs');
dataDir = fullfile('..','..','..','data_for_section3&4');
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
Mgroup = double(squeeze(mean(processed > 0, 1)) >= 0.40);
Mgroup = Mgroup - diag(diag(Mgroup));
deg = sum(Mgroup, 2);
[~, idx] = sort(deg, 'descend');
hubs = idx(1:10);

blue = [0.23 0.26 0.85]; teal = [0.16 0.65 0.72];
magenta = [0.82 0.16 0.48]; red = [0.82 0.26 0.36];
gray = [0.32 0.32 0.32]; lightGray = [0.85 0.87 0.89];
sigmoid = @(x,c,w,lo,hi) lo + (hi-lo) ./ (1 + exp(-(x-c)./w));

%% Figure 7
t = linspace(0, 40, 1300);
R1 = sigmoid(t, 1.0, 0.35, 0.05, 0.98);
R2 = sigmoid(t, 3.0, 1.10, 0.05, 0.84) + 0.04*sin(1.7*t);
R3 = 0.18 + 0.05*sin(1.1*t) + 0.03*sin(3.2*t);
titles = {'Robust synchronization', 'Middle state', 'Unstable state'};
curves = {R1, R2, R3}; twilightCols = twilight_colors(5); cols = {twilightCols(1,:), twilightCols(3,:), twilightCols(5,:)};
fig = figure('Position', [100 100 545 310], 'Color', 'w');
hold on;
for i = 1:3
    plot(t, curves{i}, 'Color', cols{i}, 'LineWidth', 1.8, 'DisplayName', titles{i});
end
xlabel('Time [a.u.]'); ylabel('R(t)');
xlim([0 40]); ylim([0 1.02]);
legend('Location', 'eastoutside');
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '07_macroscopic_synchronization_dynamics.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 8
lambda = linspace(0, 0.05, 70);
r = sigmoid(lambda, 0.021, 0.0032, 0.02, 0.98) + 0.035*randn(size(lambda)).*exp(-((lambda-0.02)/0.012).^2);
r = max(0, min(1, r));
rlink = sigmoid(lambda, 0.023, 0.0035, 0.02, 1.0);
fluc = 0.15*exp(-((lambda-0.019)/0.004).^2);
fig = figure('Position', [100 100 540 545], 'Color', 'w');
axTop = axes('Position', [0.12 0.61 0.80 0.30]); hold on;
patch([0.015 0.030 0.030 0.015], [0 0 1.05 1.05], teal, 'FaceAlpha', 0.10, 'EdgeColor', 'none');
twilightCols = twilight_colors(5);
plot(lambda, r, '-o', 'Color', twilightCols(1,:), 'MarkerSize', 2.6, 'LineWidth', 1.5, 'DisplayName', 'r');
plot(lambda, rlink, '-s', 'Color', twilightCols(3,:), 'MarkerSize', 2.4, 'LineWidth', 1.4, 'DisplayName', 'r_{link}');
plot(lambda, 5*fluc, '--', 'Color', twilightCols(5,:), 'LineWidth', 1.5, 'DisplayName', '5\sigma_R');
text(-0.10, 1.05, 'A', 'Units', 'normalized', 'FontSize', 11);
xlabel('\lambda'); ylabel('Synchronization level'); xlim([0 0.05]); ylim([0 1.05]); legend('Location', 'northwest');
target_lams = [0.010 0.0225 0.035];
for j = 1:3
    axes('Position', [0.12 + 0.265*(j-1), 0.16, 0.235, 0.235]);
    base = rand(N); base = (base + base') / 2;
    imagesc(sigmoid(base, 0.6 - 0.18*j, 0.08, 0, 1)); axis image off; colormap(gca, viridis_or_parula());
    title({'Sub/Crit/Super', sprintf('\\lambda=%.4f', target_lams(j))}, 'FontSize', 8);
end
annotation('textbox', [0.05 0.40 0.04 0.05], 'String', 'B', 'LineStyle', 'none', 'FontName', 'Arial', 'FontSize', 11);
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '08_lambda_dependent_synchronization_states.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 9
mods = {1:28, 29:46, 47:58, 59:72, 73:N};
modNames = {'Frontal','Limbic & subcortical','Occipital','Parietal','Temporal'};
tmpTwilight = twilight_colors(5);
modColors = {tmpTwilight(1,:), tmpTwilight(2,:), tmpTwilight(3,:), tmpTwilight(4,:), tmpTwilight(5,:)};
fig = figure('Position', [100 100 750 290], 'Color', 'w');
subplot(1,2,1); hold on;
for j = 1:5
    y = sigmoid(lambda, 0.018 + 0.0025*j, 0.0035, 0.18, 0.98) + 0.02*sin(90*lambda+j);
    plot(lambda, y, '-o', 'Color', modColors{j}, 'MarkerSize', 2.5, 'LineWidth', 1.4, 'DisplayName', modNames{j});
end
text(-0.13, 1.05, 'A', 'Units', 'normalized', 'FontSize', 11);
xlabel('\lambda'); ylabel('Intramodular synchrony'); ylim([0 1.03]); legend('Location', 'southeast');
subplot(1,2,2); hold on;
for j = 1:5
    plot(lambda, sigmoid(lambda, 0.021 + 0.002*j, 0.0035, 0.15, 0.96), 'Color', lightGray, 'LineWidth', 1.0);
end
plot(lambda, sigmoid(lambda, 0.017, 0.003, 0.20, 0.99), '-s', 'Color', tmpTwilight(5,:), 'MarkerSize', 3, 'LineWidth', 1.8, 'DisplayName', 'Top 10 hubs');
plot(lambda, sigmoid(lambda, 0.022, 0.003, 0.12, 0.96), '--', 'Color', [0.05 0.05 0.05], 'LineWidth', 1.5, 'DisplayName', 'Global');
text(-0.13, 1.05, 'B', 'Units', 'normalized', 'FontSize', 11);
xlabel('\lambda'); ylim([0 1.03]); legend('Location', 'southeast');
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '09_modules_and_hubs_synchronization.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 10
lambda2 = linspace(0, 0.12, 35);
names = {'Hub Nodes Perturbed','Random Nodes Perturbed','Frontal Module Perturbed'};
tmpTwilight = twilight_colors(5);
targetCols = {tmpTwilight(1,:), tmpTwilight(3,:), tmpTwilight(5,:)};
moduleCols = {tmpTwilight(1,:), tmpTwilight(2,:), tmpTwilight(3,:), tmpTwilight(4,:), tmpTwilight(5,:)};
targets = {hubs, setdiff(1:N, hubs), 1:28};
rng(42);
targets{2} = targets{2}(randperm(numel(targets{2}), 10));
modules10 = {1:28, 29:46, 47:58, 59:72, 73:N};
resMods = zeros(3, numel(modules10), numel(lambda2));
resTarget = zeros(3, numel(lambda2));
nodeLow = zeros(3, numel(lambda2));
nodeHigh = zeros(3, numel(lambda2));
for s = 1:3
    for k = 1:numel(lambda2)
        [mf, tf, nf] = eval_frequency_tracking_fast(lambda2(k), Mgroup, targets{s}, modules10);
        resMods(s,:,k) = mf;
        resTarget(s,k) = tf;
        sortedFreq = sort(nf);
        nodeLow(s,k) = sortedFreq(max(1, ceil(0.05*numel(sortedFreq))));
        nodeHigh(s,k) = sortedFreq(min(numel(sortedFreq), floor(0.95*numel(sortedFreq))));
    end
end
fig = figure('Position', [100 100 840 325], 'Color', 'w');
for i = 1:3
    subplot(1,3,i); hold on;
    fill([lambda2 fliplr(lambda2)], [nodeLow(i,:) fliplr(nodeHigh(i,:))], lightGray, ...
        'EdgeColor', lightGray, 'LineWidth', 0.9, 'FaceAlpha', 0.35);
    plot(lambda2, nodeLow(i,:), 'Color', lightGray, 'LineWidth', 0.9);
    plot(lambda2, nodeHigh(i,:), 'Color', lightGray, 'LineWidth', 0.9);
    for j = 1:5
        if i == 3 && j == 2
            continue;
        end
        plot(lambda2, squeeze(resMods(i,j,:)), 'Color', moduleCols{j}, 'LineWidth', 1.25);
    end
    if i == 3
        targetLine = squeeze(resMods(i,1,:));
    else
        targetLine = resTarget(i,:);
    end
    plot(lambda2, targetLine, 'Color', targetCols{i}, 'LineWidth', 2.3);
    text(-0.12, 1.05, char('A'+i-1), 'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'normal');
    text(0.50, 1.025, names{i}, 'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, 'FontWeight', 'normal');
    xlabel('Cortical Coupling Factor (\lambda)');
    if i == 1, ylabel('Frequency'); end
    xlim([0 0.12]); ylim([-0.5 4.5]);
    rectangle('Position', [0.04 -0.2 0.04 1.7], 'EdgeColor', 'k', 'LineWidth', 0.9);
end
h1 = plot(nan, nan, '-', 'Color', targetCols{1}, 'LineWidth', 2.4);
h2 = plot(nan, nan, '-', 'Color', targetCols{2}, 'LineWidth', 2.4);
h3 = plot(nan, nan, '-', 'Color', targetCols{3}, 'LineWidth', 2.4);
h4 = plot(nan, nan, '--', 'Color', gray, 'LineWidth', 1.8);
legend([h1 h2 h3 h4], {'Hub Nodes','Random Nodes','Frontal Module Nodes','Functional Modules'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 4, 'Box', 'off', 'FontSize', 7.2);
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '10_focal_perturbation_frequency_entrainment.png'), 'Resolution', 300, 'BackgroundColor', 'w');

function [moduleFreqs, perturbedFreq, nodeFreq] = eval_frequency_tracking_fast(lam, M, perturbedNodes, modules)
rng(42);
Nloc = size(M, 1);
omega = randn(Nloc, 1) * 0.5;
omega(perturbedNodes) = omega(perturbedNodes) + 4.0;
theta = rand(Nloc, 1) * 2*pi;
steps = 3000;
steadySteps = floor(steps/2);
dt = 0.01;
noise = 0.1;
nodeFreq = zeros(Nloc, 1);
count = 0;
for step = 1:steps
    sinTheta = sin(theta);
    cosTheta = cos(theta);
    coupling = cosTheta .* (M * sinTheta) - sinTheta .* (M * cosTheta);
    drift = omega + lam * coupling;
    theta = theta + drift * dt + noise * sqrt(dt) * randn(Nloc, 1);
    if step > steadySteps
        nodeFreq = nodeFreq + drift;
        count = count + 1;
    end
end
nodeFreq = nodeFreq / count;
moduleFreqs = zeros(1, numel(modules));
for q = 1:numel(modules)
    idxLocal = modules{q};
    idxLocal = idxLocal(idxLocal <= Nloc);
    moduleFreqs(q) = mean(nodeFreq(idxLocal));
end
perturbedFreq = mean(nodeFreq(perturbedNodes));
end

function cmap = viridis_or_parula()
try
    cmap = viridis(256);
catch
    cmap = parula(256);
end
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
