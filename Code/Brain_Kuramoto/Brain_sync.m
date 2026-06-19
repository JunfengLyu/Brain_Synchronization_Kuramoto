clear; clc; close all;

baseDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(baseDir, 'data');
figDir = fullfile(baseDir, '..', '..', 'Report', 'Figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end
rng(42);

[Mgroup, probMat] = load_aal_consensus(dataDir, 0.40);
modules = brain_modules(size(Mgroup, 1));
deg = sum(Mgroup, 2);
[~, hubOrder] = sort(deg, 'descend');
hubs = hubOrder(1:min(10, numel(hubOrder)));

make_fig07(Mgroup, figDir);
make_fig08(Mgroup, figDir);
make_fig09(Mgroup, modules, hubs, figDir);
make_fig10(Mgroup, modules, hubs, figDir);

fprintf('Brain_sync.m finished: Fig.7-10 are generated from real AAL connectivity data.\n');

function make_fig07(M, figDir)
lambdaVals = [0.040, 0.023, 0.010];
labels = {'Robust synchronization', 'Middle state', 'Unstable state'};
cols = twilight_colors(numel(lambdaVals));
fig = figure('Color', 'w', 'Position', [100 100 545 310]); hold on;
for i = 1:numel(lambdaVals)
    [t, R] = simulate_trace(M, lambdaVals(i), 1300, 0.03, 0.2, 20+i);
    plot(t, R, 'LineWidth', 1.5, 'Color', cols(i,:), 'DisplayName', labels{i});
end
xlabel('Time [a.u.]'); ylabel('$R(t)$', 'Interpreter', 'latex');
xlim([0 max(t)]); ylim([0 1.02]);
legend('Location', 'eastoutside', 'Box', 'off');
style_axes(gca);
exportgraphics(fig, fullfile(figDir, '07_macroscopic_synchronization_dynamics.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig08(M, figDir)
lambdas = linspace(0, 0.05, 70);
[Rmean, Rstd, Rlink, snaps] = scan_network(M, lambdas, 900, 480, 0.03, 0.2, 61);
cols = twilight_colors(3);
fig = figure('Color', 'w', 'Position', [100 100 540 545]);
ax = axes('Position', [0.12 0.61 0.80 0.30]); hold on;
patch([0.015 0.030 0.030 0.015], [0 0 1.05 1.05], [0.16 0.65 0.72], ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none');
plot(lambdas, Rmean, '-o', 'MarkerSize', 2.4, 'LineWidth', 1.5, 'Color', cols(1,:), 'DisplayName', '$R$');
plot(lambdas, Rlink, '-s', 'MarkerSize', 2.2, 'LineWidth', 1.5, 'Color', cols(2,:), 'DisplayName', '$R_{link}$');
plot(lambdas, 5*Rstd, '--', 'LineWidth', 1.5, 'Color', cols(3,:), 'DisplayName', '$5\sigma_R$');
text(-0.09, 1.05, 'A', 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
xlabel('$\lambda$', 'Interpreter', 'latex'); ylabel('Synchronization level');
xlim([0 0.05]); ylim([0 1.05]);
legend('Location', 'northwest', 'Box', 'off', 'Interpreter', 'latex');
style_axes(ax);

targetLams = [0.010, 0.0225, 0.035];
heatSize = 0.215;
heatY = 0.16;
heatX = [0.12 0.385 0.65];
for j = 1:3
    axj = axes('Position', [heatX(j) heatY heatSize heatSize]); %#ok<LAXES>
    [~, idx] = min(abs(lambdas - targetLams(j)));
    fc = plv_matrix(snaps{idx});
    imagesc(axj, fc, [0 1]); axis(axj, 'image'); axis(axj, 'off');
    colormap(axj, parula(256));
    if j == 1
        text(axj, -0.20, 1.12, 'B', 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    end
    text(axj, 0.03, 0.95, sprintf('$\\lambda=%.4f$', targetLams(j)), ...
        'Units', 'normalized', 'Interpreter', 'latex', 'FontName', 'Arial', 'FontSize', 8, ...
        'Color', 'w', 'VerticalAlignment', 'top');
end
cb = colorbar(axj, 'Position', [0.90 heatY 0.018 heatSize]); cb.Label.String = 'PLV';
exportgraphics(fig, fullfile(figDir, '08_lambda_dependent_synchronization_states.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig09(M, modules, hubs, figDir)
lambdas = linspace(0, 0.05, 56);
[~, ~, ~, snaps] = scan_network(M, lambdas, 720, 380, 0.03, 0.2, 74);
modNames = fieldnames(modules);
modCols = twilight_colors(numel(modNames));
vals = zeros(numel(modNames), numel(lambdas));
hubVals = zeros(1, numel(lambdas));
globalVals = zeros(1, numel(lambdas));
for k = 1:numel(lambdas)
    th = snaps{k};
    for m = 1:numel(modNames)
        vals(m,k) = local_order(th, modules.(modNames{m}));
    end
    hubVals(k) = local_order(th, hubs);
    globalVals(k) = mean(abs(mean(exp(1i*th), 2)));
end

fig = figure('Color', 'w', 'Position', [100 100 750 290]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile; hold on;
for m = 1:numel(modNames)
    plot(lambdas, vals(m,:), '-o', 'MarkerSize', 2.3, 'LineWidth', 1.5, ...
        'Color', modCols(m,:), 'DisplayName', pretty_name(modNames{m}));
end
text(-0.13, 1.05, 'A', 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
xlabel('$\lambda$', 'Interpreter', 'latex'); ylabel('Intramodular synchrony');
xlim([0 0.05]); ylim([0 1.03]);
legend('Location', 'southeast', 'Box', 'off', 'FontSize', 7);
style_axes(ax);

ax = nexttile; hold on;
for m = 1:numel(modNames)
    plot(lambdas, vals(m,:), 'LineWidth', 1.0, 'Color', [0.82 0.84 0.86]);
end
plot(lambdas, hubVals, '-s', 'MarkerSize', 2.8, 'LineWidth', 1.8, ...
    'Color', modCols(end,:), 'DisplayName', 'Top 10 hubs');
plot(lambdas, globalVals, '--', 'LineWidth', 1.5, 'Color', [0.05 0.05 0.05], 'DisplayName', 'Global');
text(-0.13, 1.05, 'B', 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
xlabel('$\lambda$', 'Interpreter', 'latex'); xlim([0 0.05]); ylim([0 1.03]);
legend('Location', 'southeast', 'Box', 'off');
style_axes(ax);
exportgraphics(fig, fullfile(figDir, '09_modules_and_hubs_synchronization.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig10(M, modules, hubs, figDir)
lambdas = linspace(0, 0.12, 35);
modNames = fieldnames(modules);
rng(42);
nonHubs = setdiff((1:size(M,1))', hubs);
randomNodes = nonHubs(randperm(numel(nonHubs), numel(hubs)));
groups = {hubs, randomNodes, modules.Frontal};
groupNames = {'Hub Nodes Perturbed', 'Random Nodes Perturbed', 'Frontal Module Perturbed'};
targetCols = twilight_colors(3);
modCols = twilight_colors(numel(modNames));
lightGray = [0.82 0.84 0.86];

fig = figure('Color', 'w', 'Position', [100 100 860 340]);
panelPos = [0.08 0.23 0.245 0.67; 0.385 0.23 0.245 0.67; 0.690 0.23 0.245 0.67];
for g = 1:3
    targetNodes = groups{g};
    modFreq = zeros(numel(modNames), numel(lambdas));
    targetFreq = zeros(1, numel(lambdas));
    nodeFreq = zeros(size(M,1), numel(lambdas));
    for k = 1:numel(lambdas)
        [modFreq(:,k), targetFreq(k), nodeFreq(:,k)] = frequency_tracking(M, lambdas(k), targetNodes, modules, 42);
    end
    low = percentile_matrix(nodeFreq, 5);
    high = percentile_matrix(nodeFreq, 95);
    ax = axes('Position', panelPos(g,:)); hold on; %#ok<LAXES>
    fill([lambdas fliplr(lambdas)], [low fliplr(high)], lightGray, ...
        'FaceAlpha', 0.35, 'EdgeColor', lightGray, 'LineWidth', 0.9);
    plot(lambdas, low, 'Color', lightGray, 'LineWidth', 0.9);
    plot(lambdas, high, 'Color', lightGray, 'LineWidth', 0.9);
    moduleHandles = gobjects(numel(modNames),1);
    for m = 1:numel(modNames)
        if g == 3 && strcmp(modNames{m}, 'Frontal'), continue; end
        moduleHandles(m) = plot(lambdas, modFreq(m,:), 'Color', modCols(m,:), 'LineWidth', 1.25);
    end
    if g == 3
        targetLine = modFreq(strcmp(modNames, 'Frontal'), :);
    else
        targetLine = targetFreq;
    end
    plot(lambdas, targetLine, 'Color', targetCols(g,:), 'LineWidth', 2.3);
    text(-0.12, 1.03, char('A'+g-1), 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    xlabel('Cortical Coupling Factor ($\lambda$)', 'Interpreter', 'latex');
    if g == 1, ylabel('Frequency'); end
    xlim([0 0.12]); ylim([-0.5 4.5]); style_axes(ax);
    xz = [0.04 0.08]; yz = [-0.20 1.50];
    rectangle('Position', [xz(1) yz(1) diff(xz) diff(yz)], 'EdgeColor', 'k', 'LineWidth', 0.9);
    axins = axes('Position', inset_position(ax, [0.60 0.56 0.34 0.33])); hold(axins, 'on'); %#ok<LAXES>
    for m = 1:numel(modNames)
        if isgraphics(moduleHandles(m))
            plot(axins, lambdas, modFreq(m,:), 'Color', modCols(m,:), 'LineWidth', 1.25);
        end
    end
    plot(axins, lambdas, targetLine, 'Color', targetCols(g,:), 'LineWidth', 2.3);
    xlim(axins, xz); ylim(axins, yz); set(axins, 'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.9);
    drawnow;
    connect_inset(ax, axins, xz, yz);
end
legAx = axes('Position', [0.22 0.02 0.58 0.05], 'Visible', 'off'); hold(legAx, 'on');
legendHandles = gobjects(1,4);
legendHandles(1) = plot(legAx, nan, nan, '-', 'Color', targetCols(1,:), 'LineWidth', 2.4);
legendHandles(2) = plot(legAx, nan, nan, '-', 'Color', targetCols(2,:), 'LineWidth', 2.4);
legendHandles(3) = plot(legAx, nan, nan, '-', 'Color', targetCols(3,:), 'LineWidth', 2.4);
legendHandles(4) = plot(legAx, nan, nan, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.8);
legend(legAx, legendHandles, {'Hub Nodes','Random Nodes','Frontal Module Nodes','Functional Modules'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 4, 'Box', 'off', 'FontSize', 7.2);
exportgraphics(fig, fullfile(figDir, '10_focal_perturbation_frequency_entrainment.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function [Mgroup, probMat] = load_aal_consensus(dataDir, threshold)
raw = load(fullfile(dataDir, 'rawdata.mat'));
SC = raw.SCmatrices;
[S, N, ~] = size(SC);
bin = zeros(S, N, N);
for s = 1:S
    A = squeeze(SC(s,:,:));
    A = (A + A') / 2;
    A(1:N+1:end) = 0;
    bin(s,:,:) = A > 0;
end
probMat = squeeze(mean(bin, 1));
Mgroup = double(probMat >= threshold);
Mgroup(1:N+1:end) = 0;
end

function [t, R] = simulate_trace(M, lambda, steps, dt, noise, seed)
rng(seed);
N = size(M,1);
deg = sum(M,2); deg(deg == 0) = 1;
W = M ./ deg;
omega = randn(N,1);
theta = rand(N,1) * 2*pi;
R = zeros(steps,1);
for s = 1:steps
    phase = theta' - theta;
    theta = theta + dt * (omega + lambda * 90 * sum(W .* sin(phase), 2)) + noise * sqrt(dt) * randn(N,1);
    R(s) = abs(mean(exp(1i*theta)));
end
t = (0:steps-1)' * dt;
end

function [Rmean, Rstd, Rlink, snaps] = scan_network(M, lambdas, steps, burn, dt, noise, seed)
rng(seed);
N = size(M,1);
deg = sum(M,2); deg(deg == 0) = 1;
W = M ./ deg;
omega = randn(N,1);
theta0 = rand(N,1) * 2*pi;
edgeDen = max(sum(M(:)), 1);
Rmean = zeros(size(lambdas)); Rstd = Rmean; Rlink = Rmean; snaps = cell(size(lambdas));
for k = 1:numel(lambdas)
    theta = theta0;
    rTrace = zeros(steps-burn, 1);
    linkTrace = rTrace;
    sample = [];
    idx = 1;
    for s = 1:steps
        phase = theta' - theta;
        theta = theta + dt * (omega + lambdas(k) * 90 * sum(W .* sin(phase), 2)) + noise * sqrt(dt) * randn(N,1);
        if s > burn
            rTrace(idx) = abs(mean(exp(1i*theta)));
            linkTrace(idx) = sum(sum(M .* cos(theta' - theta))) / edgeDen;
            if mod(idx, 8) == 1
                sample = [sample; theta']; %#ok<AGROW>
            end
            idx = idx + 1;
        end
    end
    Rmean(k) = mean(rTrace); Rstd(k) = std(rTrace); Rlink(k) = max(0, min(1, mean(linkTrace)));
    snaps{k} = sample;
end
end

function fc = plv_matrix(thetaSamples)
Z = exp(1i * thetaSamples);
fc = abs(Z' * Z) / max(1, size(Z,1));
fc(1:size(fc,1)+1:end) = 1;
end

function val = local_order(thetaSamples, nodes)
nodes = nodes(nodes >= 1 & nodes <= size(thetaSamples, 2));
val = mean(abs(mean(exp(1i * thetaSamples(:,nodes)), 2)));
end

function [modFreq, targetFreq, nodeFreq] = frequency_tracking(M, lambda, targetNodes, modules, seed)
rng(seed);
N = size(M,1);
omega = 0.5 * randn(N,1);
omega(targetNodes) = omega(targetNodes) + 4.0;
theta = rand(N,1) * 2*pi;
steps = 3000; burn = floor(steps/2); dt = 0.01; noise = 0.1;
nodeFreq = zeros(N,1); count = 0;
for s = 1:steps
    sinTheta = sin(theta); cosTheta = cos(theta);
    coupling = cosTheta .* (M * sinTheta) - sinTheta .* (M * cosTheta);
    drift = omega + lambda * coupling;
    theta = theta + drift * dt + noise * sqrt(dt) * randn(N,1);
    if s > burn
        nodeFreq = nodeFreq + drift; count = count + 1;
    end
end
nodeFreq = nodeFreq / max(count,1);
names = fieldnames(modules);
modFreq = zeros(numel(names),1);
for i = 1:numel(names)
    nodes = modules.(names{i});
    nodes = nodes(nodes <= N);
    modFreq(i) = mean(nodeFreq(nodes));
end
targetFreq = mean(nodeFreq(targetNodes));
end

function modules = brain_modules(N)
modules.Frontal = 1:min(28,N);
modules.Limbic_Subcortical = 29:min(46,N);
modules.Occipital = 47:min(58,N);
modules.Parietal = 59:min(72,N);
modules.Temporal = 73:N;
end

function name = pretty_name(s)
name = strrep(s, '_', ' & ');
end

function style_axes(ax)
set(ax, 'FontName', 'Arial', 'FontSize', 8.5, 'LineWidth', 1.1, 'TickDir', 'out', 'Box', 'off');
grid(ax, 'off');
end

function cols = twilight_colors(n)
base = [0.8858 0.8500 0.8879; 0.6443 0.4390 0.7389; 0.1849 0.0794 0.2131; ...
        0.2453 0.2878 0.5373; 0.2890 0.5650 0.7595; 0.7590 0.7060 0.5680; ...
        0.8858 0.8500 0.8879];
x = linspace(0,1,size(base,1)); xi = linspace(0.08,0.88,n);
cols = interp1(x, base, xi);
end

function q = percentile_matrix(X, pct)
q = zeros(1, size(X,2));
for c = 1:size(X,2)
    v = sort(X(:,c));
    pos = 1 + (numel(v)-1) * pct/100;
    lo = floor(pos); hi = ceil(pos);
    if lo == hi
        q(c) = v(lo);
    else
        q(c) = v(lo) + (pos-lo) * (v(hi)-v(lo));
    end
end
end

function pos = inset_position(ax, rel)
outer = get(ax, 'Position');
pos = [outer(1)+rel(1)*outer(3), outer(2)+rel(2)*outer(4), rel(3)*outer(3), rel(4)*outer(4)];
end

function connect_inset(ax, axins, xz, yz)
p1 = data_to_fig(ax, [xz(1) yz(1)]);
p2 = data_to_fig(ax, [xz(2) yz(1)]);
ip = get(axins, 'Position');
annotation('line', [p1(1) ip(1)], [p1(2) ip(2)], 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
annotation('line', [p2(1) ip(1)+ip(3)], [p2(2) ip(2)], 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
end

function p = data_to_fig(ax, xy)
oldUnits = get(ax, 'Units'); set(ax, 'Units', 'normalized');
axpos = get(ax, 'Position'); set(ax, 'Units', oldUnits);
xl = xlim(ax); yl = ylim(ax);
p = [axpos(1) + (xy(1)-xl(1))/diff(xl)*axpos(3), axpos(2) + (xy(2)-yl(1))/diff(yl)*axpos(4)];
end
