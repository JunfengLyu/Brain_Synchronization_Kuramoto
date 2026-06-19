%% Synchronization Transition
%% Mean-field bifurcation and finite-size effects

clear;clc;close all;

sigma = 0.5;

Kc = critical_k(sigma);

K_left  = 0.75*Kc;
K_crit  = Kc;
K_right = 1.35*Kc;

%% Mean-field curves
R = linspace(0,1,800);
psi_left  = psi_curve(K_left ,R,sigma);
psi_crit  = psi_curve(K_crit ,R,sigma);
psi_right = psi_curve(K_right,R,sigma);

%% Finite-size simulations
K_values = linspace(0,2,41);
N_list = [10 100 1000 10000];
R_mean = cell(length(N_list),1);
R_std  = cell(length(N_list),1);

for i = 1:length(N_list)
    [R_mean{i},R_std{i}] = simulate_kuramoto_sweep( ...
        N_list(i), ...
        K_values, ...
        sigma, ...
        0.05, ...
        150, ...
        75, ...
        1);

end

%% Mean-field prediction
K_dense = linspace(0,2,300);
R_mf = zeros(size(K_dense));
for i = 1:length(K_dense)
    R_mf(i) = mean_field_r(K_dense(i),sigma);
end

%% Plot
fig = figure( ...
    'Color','w', ...
    'Position',[100 100 1150 430], ...
    'Toolbar','none');

blues = [
247 251 255;
222 235 247;
198 219 239;
158 202 225;
107 174 214;
66 146 198;
33 113 181;
8 81 156;
8 48 107]/255;

%% Stable fixed point
idx0 = find(psi_right(1:end-1).*psi_right(2:end)<=0,1,'last');
Rstar = interp1( ...
    psi_right(idx0:idx0+1), ...
    R(idx0:idx0+1), ...
    0);

%% Panel A

ax1 = subplot(1,2,1);
hold on;

plot(psi_left,R,...
    'Color',blues(6,:),...
    'LineWidth',2.5);

plot(psi_crit,R,...
    'Color',blues(8,:),...
    'LineWidth',2.5);

plot(psi_right,R,...
    'Color',[0.82 0.25 0.45],...
    'LineWidth',2.5);

plot([0 0],[0 1.05],...
    'k',...
    'LineWidth',1.5);

scatter(0,Rstar,...
    80,...
    [0.82 0.25 0.45],...
    'filled');

text(0.06,Rstar+0.03,...
    '$R^\ast$',...
    'Interpreter','latex',...
    'FontSize',12,...
    'Color',[0.82 0.25 0.45]);

xlabel('$\psi(K;R)$',...
    'Interpreter','latex');

ylabel('$R$',...
    'Interpreter','latex');

xmin = min([psi_left(:);psi_crit(:);psi_right(:)]);
xmax = max([psi_left(:);psi_crit(:);psi_right(:)]);

xlim([1.05*xmin 1.05*xmax]);
ylim([0 1.05]);

legend({ ...
    '$K<K_c$', ...
    '$K=K_c$', ...
    '$K>K_c$'},...
    'Interpreter','latex',...
    'Box','off');

set(gca,...
    'FontSize',11,...
    'LineWidth',1,...
    'TickDir','out');

box off;

text(-0.12,1.05,'A',...
    'Units','normalized',...
    'FontSize',18);

%% Panel B

ax2 = subplot(1,2,2);
hold on;

plot(K_dense,R_mf,...
    'Color',[0.55 0.55 0.55],...
    'LineWidth',3,...
    'DisplayName','MF prediction');

cols = [ ...
0.30 0.75 0.85;
0.25 0.25 0.85;
0.75 0.25 0.80;
0.85 0.20 0.35];

for i = 1:length(N_list)

    errorbar(K_values,...
        R_mean{i},...
        R_std{i},...
        'o',...
        'Color',cols(i,:),...
        'MarkerFaceColor',cols(i,:),...
        'MarkerSize',5,...
        'CapSize',2,...
        'LineWidth',0.6,...
        'DisplayName',sprintf('$N=%d$',N_list(i)));

end

xlabel('$K$',...
    'Interpreter','latex');

ylabel('$R$',...
    'Interpreter','latex');

xlim([0 2]);
ylim([0 1.05]);

legend( ...
    'Interpreter','latex',...
    'Location','southeast',...
    'Box','off');

set(gca,...
    'FontSize',11,...
    'LineWidth',1,...
    'TickDir','out');

box off;

text(-0.12,1.05,'B',...
    'Units','normalized',...
    'FontSize',18);

%% Shared horizontal guide
drawnow;
pos1 = ax1.Position;
xfig = pos1(1) + ...
    (0-ax1.XLim(1))/diff(ax1.XLim)*pos1(3);
yfig = pos1(2) + ...
    (Rstar-ax1.YLim(1))/diff(ax1.YLim)*pos1(4);
annotation(fig,...
    'line',...
    [0.08 0.94],...
    [yfig yfig],...
    'LineStyle','--',...
    'Color',[0.70 0.70 0.70],...
    'LineWidth',1.3);
annotation(fig,...
    'textarrow',...
    [xfig+0.10 xfig],...
    [yfig+0.10 yfig],...
    'String','$R^\ast(K>K_c)$',...
    'Interpreter','latex',...
    'FontSize',11);

%% Functions

function g = gaussian(x,sigma)
g = exp(-(x.^2)/(2*sigma^2))./(sqrt(2*pi)*sigma);
end

function kc = critical_k(sigma)
kc = 2/(pi*gaussian(0,sigma));
end

function psi = psi_curve(K,r,sigma)

x = linspace(-1,1,2501);

kernel = sqrt(max(0,1-x.^2));

values = gaussian(K*r(:).*x,sigma).*kernel;

integral = trapz(x,values,2);

psi = K*integral - 1;

end

function rstar = mean_field_r(K,sigma)

kc = critical_k(sigma);

if K<=kc
    rstar = 0;
    return
end

grid = linspace(1e-4,0.999,800);

vals = psi_curve(K,grid,sigma);

idx = find(vals(1:end-1).*vals(2:end)<=0,1,'last');

if isempty(idx)
    [~,idx] = min(abs(vals));
    rstar = grid(idx);
    return
end

lo = grid(idx);
hi = grid(idx+1);

for k = 1:40

    mid = 0.5*(lo+hi);

    f1 = psi_curve(K,lo,sigma);
    f2 = psi_curve(K,mid,sigma);

    if f1*f2<=0
        hi = mid;
    else
        lo = mid;
    end

end

rstar = 0.5*(lo+hi);

end

function [r_mean,r_std] = simulate_kuramoto_sweep(n,K_values,sigma,dt,T,burn,seed)

rng(seed);

omega = 1 + sigma*randn(n,1);

theta = 2*pi*rand(length(K_values),n);

Kcol = K_values(:);

steps = round(T/dt);
burn_steps = round(burn/dt);

r_sum = zeros(length(K_values),1);
r_sq  = zeros(length(K_values),1);

count = 0;

for step = 1:steps

    c = mean(cos(theta),2);
    s = mean(sin(theta),2);

    r = hypot(c,s);
    psi = atan2(s,c);

    theta = theta + dt*( ...
        omega' + ...
        Kcol.*r.*sin(psi-theta));

    if step>burn_steps

        r_sum = r_sum + r;
        r_sq  = r_sq + r.^2;

        count = count + 1;

    end

end

r_mean = r_sum/count;
r_std  = sqrt(max(0,r_sq/count-r_mean.^2));

end