clear; clc; close all;

baseDir = fileparts(mfilename('fullpath'));
figDir = fullfile(baseDir, '..', '..', 'Report', 'Figs');
if ~exist(figDir, 'dir'), mkdir(figDir); end

make_fig15(figDir);
make_fig16(figDir);
make_fig17(figDir);

fprintf('Brain_circadian.m finished: Fig.15-17 are generated.\n');

function make_fig15(figDir)
delta = 1.0;
f = 3.5 * delta;
omega = 1.4 * delta;
regimes = [10.0, 4.5] * delta;
regimeLabels = {'$K=10\Delta$', '$K=4.5\Delta$'};
shifts = [-8.5, -9.5, 9.0, 12.0];
cols = twilight_colors(numel(shifts));
fig = figure('Color', 'w', 'Position', [100 100 720 335]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
thetaCircle = linspace(0, 2*pi, 500);
for r = 1:2
    ax = nexttile; hold on;
    k = regimes(r);
    zst = stable_point_fast(k, f, delta, omega);
    pts = fixed_points_all(k, f, delta, omega);
    plot(cos(thetaCircle), sin(thetaCircle), '--', 'LineWidth', 1.0, 'Color', [0.82 0.84 0.86]);
    plot([-1 1], [0 0], 'Color', [0.82 0.84 0.86], 'LineWidth', 0.9);
    plot([0 0], [-1 1], 'Color', [0.82 0.84 0.86], 'LineWidth', 0.9);
    for rad = [0.25 0.55 0.85]
        for ang = linspace(0, 2*pi, 18)
            z0 = rad * exp(1i*ang);
            [~, z] = trajectory(z0, 25, k, f, delta, omega, 1200, 0.035);
            plot(real(z), imag(z), 'Color', [0.82 0.84 0.86], 'LineWidth', 0.55);
        end
    end
    for s = 1:numel(shifts)
        z0 = zst * exp(1i * shifts(s) * 2*pi/24);
        [~, z] = trajectory(z0, 80, k, f, delta, omega, 2600, 0.035);
        plot(real(z), imag(z), 'LineWidth', 1.5, 'Color', cols(s,:));
        plot(real(z(1)), imag(z(1)), 'o', 'MarkerSize', 3.2, 'MarkerFaceColor', cols(s,:), 'MarkerEdgeColor', cols(s,:));
    end
    for p = 1:size(pts,1)
        z = pts(p,1) + 1i*pts(p,2);
        ev = eig(jacobian(z, k, f, delta, omega));
        if all(real(ev) < 0)
            plot(real(z), imag(z), 'p', 'MarkerSize', 7, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
        elseif all(real(ev) > 0)
            plot(real(z), imag(z), 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
        else
            plot(real(z), imag(z), '+', 'MarkerSize', 8, 'LineWidth', 1.5, 'Color', 'k');
        end
    end
    text(-0.13, 1.04, char('A'+r-1), 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    text(0.05, 0.92, regimeLabels{r}, 'Units', 'normalized', 'Interpreter', 'latex', 'FontName', 'Arial', 'FontSize', 9);
    axis equal; xlim([-1.02 1.02]); ylim([-1.02 1.02]);
    xlabel('$\operatorname{Re}(z)$', 'Interpreter', 'latex');
    if r == 1, ylabel('$\operatorname{Im}(z)$', 'Interpreter', 'latex'); end
    style_axes(ax);
end
exportgraphics(fig, fullfile(figDir, '15_circadian_phase_space_dynamics.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig16(figDir)
delta = 3.8e-3;
k = 4.5 * delta; f = 3.5 * delta; omega = 1.4 * delta;
zst = stable_point_fast(k, f, delta, omega);
cases = [-3 -6 -9 12 3 6 9];
labels = {'3 E','6 E','9 E','12 E/W','3 W','6 W','9 W'};
styles = {'-','-','-','-','--','--','--'};
cols = twilight_colors(4);
caseCols = [cols(1,:); cols(2,:); cols(3,:); cols(4,:); cols(1,:); cols(2,:); cols(3,:)];
fig = figure('Color', 'w', 'Position', [100 100 470 335]); hold on;
for i = 1:numel(cases)
    z0 = zst * exp(1i * cases(i) * 2*pi/24);
    [t, z] = trajectory(z0, 14*24, k, f, delta, omega, 900, inf);
    d = abs(z - zst);
    stop = find(d <= 0.2, 1, 'first');
    if isempty(stop), stop = numel(d); end
    plot(t(1:stop)/24, d(1:stop), styles{i}, 'LineWidth', 1.6, 'Color', caseCols(i,:), 'DisplayName', labels{i});
end
yline(0.2, 'k-', 'LineWidth', 1.1);
xlabel('Days'); ylabel('$|z(t)-z_{st}|$', 'Interpreter', 'latex');
xlim([0 14]); ylim([0 2.0]);
legend('Location', 'northeast', 'NumColumns', 2, 'Box', 'off');
style_axes(gca);
exportgraphics(fig, fullfile(figDir, '16_circadian_recovery_trajectories.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function make_fig17(figDir)
delta = 3.8e-3;
kRef = 4.5 * delta; fRef = 3.5 * delta; omegaRef = 1.4 * delta;
cases = [-3 -6 -9 12 3 6 9];
labels = {'3 E','6 E','9 E','12 E/W','3 W','6 W','9 W'};
styles = {'-','-','-','-','--','--','--'};
cols = twilight_colors(4);
caseCols = [cols(1,:); cols(2,:); cols(3,:); cols(4,:); cols(1,:); cols(2,:); cols(3,:)];
scans = {
    '$K/\Delta$', linspace(2.0, 15.0, 72), [2 24], 1;
    '$F/\Delta$', linspace(1.5, 5.8, 70), [0 45], 2;
    '$\Omega/\Delta$', linspace(-3.7, 3.7, 78), [0 34], 3
};
fig = figure('Color', 'w', 'Position', [100 100 820 280]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:3
    ax = nexttile; hold on;
    xs = scans{s,2};
    for c = 1:numel(cases)
        rec = nan(size(xs));
        for i = 1:numel(xs)
            if scans{s,4} == 1
                k = xs(i)*delta; f = fRef; omega = omegaRef;
            elseif scans{s,4} == 2
                k = kRef; f = xs(i)*delta; omega = omegaRef;
            else
                k = kRef; f = fRef; omega = xs(i)*delta;
            end
            zst = stable_point_fast(k, f, delta, omega);
            rec(i) = recovery_time(zst, cases(c), k, f, delta, omega, 0.2, 28);
        end
        plot(xs, rec, styles{c}, 'LineWidth', 1.25, 'Color', caseCols(c,:), 'DisplayName', labels{c});
    end
    text(-0.15, 1.04, char('A'+s-1), 'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 11);
    xlabel(scans{s,1}, 'Interpreter', 'latex');
    if s == 1, ylabel('Recovery time (days)'); end
    ylim(scans{s,3});
    if s == 2
        legend('Location', 'northeast', 'NumColumns', 2, 'Box', 'off', 'FontSize', 6.4);
    end
    style_axes(ax);
end
exportgraphics(fig, fullfile(figDir, '17_circadian_parameter_dependence.png'), 'Resolution', 300, 'BackgroundColor', 'w');
end

function dz = dzdt_complex(z, k, f, delta, omega)
dz = 0.5 * ((k*z + f) - z.^2 .* (k*conj(z) + f)) - (delta + 1i*omega) * z;
end

function dy = rhs_real(~, y, k, f, delta, omega)
z = y(1) + 1i*y(2);
dz = dzdt_complex(z, k, f, delta, omega);
dy = [real(dz); imag(dz)];
end

function z = stable_point_fast(k, f, delta, omega)
starts = [0.5 0; 0 0; -0.5 0; 0 0.5; 0 -0.5; 0.8 0.2; -0.2 0.8];
best = starts(1,:); bestVal = inf;
opts = optimset('Display', 'off', 'TolX', 1e-10, 'TolFun', 1e-12, 'MaxIter', 800);
for i = 1:size(starts,1)
    y = fminsearch(@(yy) rhs_norm(yy, k, f, delta, omega), starts(i,:), opts);
    val = rhs_norm(y, k, f, delta, omega);
    if norm(y) <= 1.02 && val < bestVal
        best = y; bestVal = val;
    end
end
z = best(1) + 1i*best(2);
end

function pts = fixed_points_all(k, f, delta, omega)
grid = linspace(-0.92, 0.92, 11);
pts = [];
opts = optimset('Display', 'off', 'TolX', 1e-10, 'TolFun', 1e-12, 'MaxIter', 600);
for x = grid
    for y = grid
        if x^2 + y^2 > 1, continue; end
        p = fminsearch(@(yy) rhs_norm(yy, k, f, delta, omega), [x y], opts);
        if norm(p) <= 1.02 && rhs_norm(p, k, f, delta, omega) < 1e-9
            if isempty(pts) || all(sqrt(sum((pts - p).^2, 2)) > 1e-4)
                pts = [pts; p]; %#ok<AGROW>
            end
        end
    end
end
if isempty(pts)
    z = stable_point_fast(k, f, delta, omega);
    pts = [real(z) imag(z)];
end
end

function val = rhs_norm(y, k, f, delta, omega)
if sum(y.^2) > 1.08
    val = 1e3 + sum(y.^2);
else
    r = rhs_real(0, y(:), k, f, delta, omega);
    val = sum(r.^2);
end
end

function J = jacobian(z, k, f, delta, omega)
epsv = 1e-5; y = [real(z); imag(z)]; J = zeros(2);
for i = 1:2
    e = zeros(2,1); e(i) = epsv;
    J(:,i) = (rhs_real(0, y+e, k, f, delta, omega) - rhs_real(0, y-e, k, f, delta, omega)) / (2*epsv);
end
end

function [t, z] = trajectory(z0, tEnd, k, f, delta, omega, n, maxStep)
if isinf(maxStep)
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
else
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', maxStep);
end
tspan = linspace(0, tEnd, n);
[t, y] = ode45(@(tt, yy) rhs_real(tt, yy, k, f, delta, omega), tspan, [real(z0); imag(z0)], opts);
z = y(:,1) + 1i*y(:,2);
end

function days = recovery_time(zst, shiftHours, k, f, delta, omega, threshold, maxDays)
z0 = zst * exp(1i * shiftHours * 2*pi/24);
[t, z] = trajectory(z0, 24*maxDays, k, f, delta, omega, 850, inf);
d = abs(z - zst);
idx = find(d <= threshold, 1, 'first');
if isempty(idx), days = nan; else, days = t(idx) / 24; end
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
