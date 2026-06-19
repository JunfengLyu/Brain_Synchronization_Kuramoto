clear; clc; close all;

%% Parameters
sigma = 0.5;
omega0 = 1.0;

N_list = [10, 100, 1000, 10000];
K_list = linspace(0, 2.0, 41);

dt = 0.05;
T_total = 500;
T_burn = 250;

nt = round(T_total / dt);
nburn = round(T_burn / dt);

rng(1);

colors = web_colormap(length(N_list));

%% Mean-field prediction
K_mf = linspace(0, 2.0, 500);
r_mf = mean_field_r_curve(K_mf, sigma);

%% Simulation
r_sim = zeros(length(N_list), length(K_list));
r_err = zeros(length(N_list), length(K_list));

for a = 1:length(N_list)
    N = N_list(a);

    omega = omega0 + sigma * randn(1,N);
    theta = 2*pi*rand(length(K_list),N);
    K_col = K_list(:);

    r_sum = zeros(length(K_list),1);
    r_sq_sum = zeros(length(K_list),1);
    count = 0;

    for n = 1:nt
        z = mean(exp(1i * theta),2);
        r = abs(z);
        psi = angle(z);

        dtheta = omega + K_col .* r .* sin(psi - theta);
        theta = theta + dt * dtheta;

        if n > nburn
            r_sum = r_sum + r;
            r_sq_sum = r_sq_sum + r.^2;
            count = count + 1;
        end
    end

    r_mean = r_sum / count;
    r_std = sqrt(max(0, r_sq_sum / count - r_mean.^2));

    r_sim(a,:) = r_mean;
    r_err(a,:) = r_std;
end

%% Plot
figure('Color','w','Position',[200 200 620 480]);

hMF = plot(K_mf, r_mf, '-', ...
    'Color',[0.45 0.45 0.45], ...
    'LineWidth',2.4);
hold on;

hSim = gobjects(length(N_list),1);
for a = 1:length(N_list)
    hSim(a) = errorbar(K_list, r_sim(a,:), r_err(a,:), ...
        'o', ...
        'Color',colors(a,:), ...
        'MarkerFaceColor',colors(a,:), ...
        'MarkerEdgeColor','k', ...
        'MarkerSize',5.2, ...
        'LineWidth',0.65, ...
        'CapSize',2.0, ...
        'LineStyle','none');
end

xlabel('K');
ylabel('R');
title('Kuramoto transition');

legend([hMF; hSim], ...
    [{'MF prediction'}, arrayfun(@(N) sprintf('N = %d',N), N_list, 'UniformOutput', false)], ...
    'Location','southeast', ...
    'Box','off');

xlim([0 2.0]);
ylim([0 1.02]);
grid off;
box off;
set(gca,'FontName','Arial','LineWidth',1.2,'TickDir','out');

%% Mean-field curve
function r_curve = mean_field_r_curve(K_list, sigma)
    r_curve = zeros(size(K_list));

    Kc = 2 * sqrt(2/pi) * sigma;

    for kk = 1:length(K_list)
        K = K_list(kk);

        if K <= Kc
            r_curve(kk) = 0;
        else
            fun = @(r) self_consistency_residual(r, K, sigma);

            try
                r_curve(kk) = fzero(fun, [1e-6, 0.999999]);
            catch
                r_grid = linspace(1e-4, 0.999, 500);
                vals = arrayfun(@(r) abs(fun(r)), r_grid);
                [~, id] = min(vals);
                r_curve(kk) = r_grid(id);
            end
        end
    end
end

%% Self-consistency equation
function F = self_consistency_residual(r, K, sigma)
    x = linspace(-1, 1, 3001);

    g = @(w) 1/(sqrt(2*pi)*sigma) * exp(-w.^2/(2*sigma^2));

    integrand = sqrt(1 - x.^2) .* g(K*r*x);

    rhs = K * trapz(x, integrand);

    F = 1 - rhs;
end

%% Web colormap
function colors = web_colormap(n)
    colors = zeros(n,3);

    for ii = 1:n
        hue = (190 + 155 * (ii - 1) / max(1,n - 1)) / 360;
        colors(ii,:) = hsl_to_rgb(hue,0.64,0.48);
    end
end

function rgb = hsl_to_rgb(h,s,l)
    c = (1 - abs(2*l - 1)) * s;
    hp = 6 * h;
    x = c * (1 - abs(mod(hp,2) - 1));

    if hp < 1
        rgb1 = [c x 0];
    elseif hp < 2
        rgb1 = [x c 0];
    elseif hp < 3
        rgb1 = [0 c x];
    elseif hp < 4
        rgb1 = [0 x c];
    elseif hp < 5
        rgb1 = [x 0 c];
    else
        rgb1 = [c 0 x];
    end

    m = l - c/2;
    rgb = rgb1 + m;
end
