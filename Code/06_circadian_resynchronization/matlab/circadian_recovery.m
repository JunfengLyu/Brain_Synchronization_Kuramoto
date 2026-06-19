clear; clc; close all;
addpath('../../common/matlab');
figDir = fullfile('..','..','..','Report','Figs');
twilightCols = twilight_colors(4);
blue = twilightCols(1,:); green = twilightCols(2,:);
red = twilightCols(3,:); magenta = twilightCols(4,:);
lightGray = [0.85 0.87 0.89]; black = [0.05 0.05 0.05];

%% Figure 15: phase-space dynamics
Delta = 1.0; F = 3.5 * Delta; Omega = 1.4 * Delta;
regimes = [10.0, 4.5] * Delta;
regimeLabels = {'K=10\Delta', 'K=4.5\Delta'};
shifts = [-8.5, -9.5, 9.0, 12.0];
shiftLabels = {'8.5 h eastward','9.5 h eastward','9.0 h westward','12 h E/W'};
shiftColors = {green, red, blue, magenta};
fig = figure('Position', [100 100 720 335], 'Color', 'w');
theta = linspace(0, 2*pi, 500);
for p = 1:2
    K = regimes(p);
    [zst, pts, types] = stable_fixed_point(K, F, Delta, Omega);
    subplot(1,2,p); hold on; axis equal;
    plot(cos(theta), sin(theta), '--', 'Color', lightGray, 'LineWidth', 1.0);
    plot([-1 1], [0 0], 'Color', lightGray, 'LineWidth', 0.9);
    plot([0 0], [-1 1], 'Color', lightGray, 'LineWidth', 0.9);
    for s = 1:numel(shifts)
        z0 = zst * exp(1i * shifts(s) * 2*pi/24);
        [~, ztraj] = simulate_traj(z0, [0 80], K, F, Delta, Omega);
        plot(real(ztraj), imag(ztraj), 'Color', shiftColors{s}, 'LineWidth', 1.5, 'DisplayName', shiftLabels{s});
        plot(real(ztraj(1)), imag(ztraj(1)), 'o', 'MarkerSize', 3.2, 'MarkerFaceColor', shiftColors{s}, 'MarkerEdgeColor', shiftColors{s});
    end
    for i = 1:numel(types)
        switch types{i}
            case 'stable'
                plot(pts(i,1), pts(i,2), 'p', 'MarkerSize', 7, 'MarkerFaceColor', black, 'MarkerEdgeColor', black);
            case 'unstable'
                plot(pts(i,1), pts(i,2), 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', black);
            otherwise
                plot(pts(i,1), pts(i,2), '+', 'MarkerSize', 8, 'LineWidth', 1.5, 'Color', black);
        end
    end
    text(-0.13, 1.04, char('A'+p-1), 'Units', 'normalized', 'FontSize', 11);
    text(0.05, 0.92, regimeLabels{p}, 'Units', 'normalized', 'FontSize', 9);
    xlabel('Re(z)'); if p == 1, ylabel('Im(z)'); end
    xlim([-1.02 1.02]); ylim([-1.02 1.02]);
end
legend('Location', 'southoutside', 'NumColumns', 4);
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '15_circadian_phase_space_dynamics.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 16: recovery trajectories
Delta = 3.8e-3; K = 4.5*Delta; F = 3.5*Delta; Omega = 1.4*Delta;
[zst, ~, ~] = stable_fixed_point(K, F, Delta, Omega);
cases = [-3 -6 -9 12 3 6 9];
labels = {'3 E','6 E','9 E','12 E/W','3 W','6 W','9 W'};
cols = {blue, red, green, magenta, blue, red, green};
styles = {'-','-','-','-','--','--','--'};
fig = figure('Position', [100 100 470 335], 'Color', 'w'); hold on;
for i = 1:numel(cases)
    z0 = zst * exp(1i * cases(i) * 2*pi/24);
    [t, ztraj] = simulate_traj(z0, [0 14*24], K, F, Delta, Omega);
    dist = abs(ztraj - zst);
    idx = find(dist <= 0.2, 1, 'first');
    if isempty(idx), idx = numel(t); end
    plot(t(1:idx)/24, dist(1:idx), styles{i}, 'Color', cols{i}, 'LineWidth', 1.6, 'DisplayName', labels{i});
end
yline(0.2, 'Color', black, 'LineWidth', 1.1);
xlabel('Days'); ylabel('|z(t)-z_{st}|');
xlim([0 14]); ylim([0 2]); legend('Location', 'northeast', 'NumColumns', 2);
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '16_circadian_recovery_trajectories.png'), 'Resolution', 300, 'BackgroundColor', 'w');

%% Figure 17: parameter dependence
Delta = 3.8e-3; Kref = 4.5*Delta; Fref = 3.5*Delta; Oref = 1.4*Delta;
scans = {linspace(2,15,40), linspace(1.5,5.8,40), linspace(-3.7,3.7,44)};
xlabels = {'K/\Delta','F/\Delta','\Omega/\Delta'};
ylims = {[2 24], [0 45], [0 34]};
fig = figure('Position', [100 100 820 280], 'Color', 'w');
for p = 1:3
    subplot(1,3,p); hold on;
    xs = scans{p};
    for c = 1:numel(cases)
        rec = nan(size(xs));
        for j = 1:numel(xs)
            switch p
                case 1
                    K = xs(j)*Delta; F = Fref; Omega = Oref;
                case 2
                    K = Kref; F = xs(j)*Delta; Omega = Oref;
                otherwise
                    K = Kref; F = Fref; Omega = xs(j)*Delta;
            end
            [zst, ~, ~] = stable_fixed_point(K, F, Delta, Omega);
            rec(j) = recovery_time(zst, cases(c), K, F, Delta, Omega);
        end
        plot(xs, rec, styles{c}, 'Color', cols{c}, 'LineWidth', 1.25, 'DisplayName', labels{c});
    end
    text(-0.15, 1.04, char('A'+p-1), 'Units', 'normalized', 'FontSize', 11);
    xlabel(xlabels{p}); if p == 1, ylabel('Recovery time (days)'); end
    ylim(ylims{p});
    if p == 2, legend('Location', 'northeast', 'NumColumns', 2, 'FontSize', 6.4); end
end
apply_kuramoto_style(fig);
exportgraphics(fig, fullfile(figDir, '17_circadian_parameter_dependence.png'), 'Resolution', 300, 'BackgroundColor', 'w');

function dz = complex_ode(z, K, F, Delta, Omega)
dz = 0.5*((K*z + F) - z.^2 .* (K*conj(z) + F)) - (Delta + 1i*Omega).*z;
end

function dy = ode_real(~, y, K, F, Delta, Omega)
z = y(1) + 1i*y(2);
dz = complex_ode(z, K, F, Delta, Omega);
dy = [real(dz); imag(dz)];
end

function [t, ztraj] = simulate_traj(z0, tspan, K, F, Delta, Omega)
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 0.02);
if numel(tspan) == 2
    tspan = linspace(tspan(1), tspan(2), max(2500, ceil((tspan(2)-tspan(1))/0.02)+1));
end
[t, Y] = ode45(@(t,y) ode_real(t, y, K, F, Delta, Omega), tspan, [real(z0); imag(z0)], opts);
ztraj = Y(:,1) + 1i*Y(:,2);
end

function [zst, pts, types] = stable_fixed_point(K, F, Delta, Omega)
grid = linspace(-0.92, 0.92, 13);
pts = [];
opts = optimoptions('fsolve', 'Display', 'off', 'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12);
for x = grid
    for y = grid
        if x^2 + y^2 <= 1
            try
                root = fsolve(@(yy) ode_real(0, yy, K, F, Delta, Omega), [x;y], opts);
                z = root(1) + 1i*root(2);
                if abs(z) <= 1.01
                    if isempty(pts) || all(vecnorm(pts - root', 2, 2) > 1e-4)
                        pts = [pts; root']; %#ok<AGROW>
                    end
                end
            catch
            end
        end
    end
end
types = cell(size(pts,1),1);
zst = pts(1,1) + 1i*pts(1,2);
for i = 1:size(pts,1)
    z = pts(i,1) + 1i*pts(i,2);
    J = numerical_jacobian(z, K, F, Delta, Omega);
    ev = eig(J);
    if all(real(ev) < 0)
        types{i} = 'stable';
        zst = z;
    elseif all(real(ev) > 0)
        types{i} = 'unstable';
    else
        types{i} = 'saddle';
    end
end
end

function J = numerical_jacobian(z, K, F, Delta, Omega)
epsv = 1e-5;
y = [real(z); imag(z)];
J = zeros(2,2);
for i = 1:2
    e = zeros(2,1); e(i) = epsv;
    J(:,i) = (ode_real(0, y+e, K, F, Delta, Omega) - ode_real(0, y-e, K, F, Delta, Omega)) / (2*epsv);
end
end

function rt = recovery_time(zst, shiftHours, K, F, Delta, Omega)
z0 = zst * exp(1i * shiftHours * 2*pi/24);
[t, ztraj] = simulate_traj(z0, [0 28*24], K, F, Delta, Omega);
dist = abs(ztraj - zst);
idx = find(dist <= 0.2, 1, 'first');
if isempty(idx)
    rt = NaN;
else
    rt = t(idx) / 24;
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
