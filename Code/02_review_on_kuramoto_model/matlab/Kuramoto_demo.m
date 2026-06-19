clear; clc; close all;

%% Parameters
N = 64;
sigma0 = 0.50;
K0 = 1.20;

dt = 0.05;
steps_per_frame = 15;
T_window = 120;
record_window = 25;

rng(1);

%% Figure
fig = figure('Color','w','Position',[100 100 1200 760]);

ax1 = subplot(2,2,1);
ax2 = subplot(2,2,2);
ax3 = subplot(2,2,[3 4]);

colormap(fig,web_colormap(N));

%% UI
uicontrol('Style','text','Units','normalized', ...
    'Position',[0.08 0.965 0.10 0.025], ...
    'String','sigma','BackgroundColor','w');

sliderSigma = uicontrol('Style','slider','Units','normalized', ...
    'Position',[0.17 0.965 0.22 0.025], ...
    'Min',0.05,'Max',0.80,'Value',sigma0);

textSigma = uicontrol('Style','text','Units','normalized', ...
    'Position',[0.40 0.965 0.08 0.025], ...
    'String',sprintf('%.2f',sigma0),'BackgroundColor','w');

uicontrol('Style','text','Units','normalized', ...
    'Position',[0.52 0.965 0.06 0.025], ...
    'String','K','BackgroundColor','w');

sliderK = uicontrol('Style','slider','Units','normalized', ...
    'Position',[0.58 0.965 0.22 0.025], ...
    'Min',0,'Max',2.0,'Value',K0);

textK = uicontrol('Style','text','Units','normalized', ...
    'Position',[0.81 0.965 0.08 0.025], ...
    'String',sprintf('%.2f',K0),'BackgroundColor','w');

%% State
state.N = N;
state.sigma = sigma0;
state.K = K0;

state.last_sigma = sigma0;
state.last_K = K0;

state.omega = [];
state.theta = [];

state.t = 0;
state.step = 0;

state.r_history = [];
state.spike_t = [];
state.spike_i = [];

% Current live point
state.live_K = K0;
state.live_r = NaN;

state.need_new_omega = true;
state.need_reset = true;

setappdata(fig,'state',state);

%% Callbacks
set(sliderSigma,'Callback',@(src,evt) slider_callback(fig, sliderSigma, sliderK, textSigma, textK));
set(sliderK,'Callback',@(src,evt) slider_callback(fig, sliderSigma, sliderK, textSigma, textK));

%% Main loop
while ishandle(fig)
    state = getappdata(fig,'state');

    if state.need_reset
        state = initialize_system(state);
        state.need_reset = false;
        setappdata(fig,'state',state);
        draw_all(ax1, ax2, ax3, state, T_window);
    end

    for q = 1:steps_per_frame
        state = step_system(state, dt, record_window);
    end

    setappdata(fig,'state',state);

    draw_all(ax1, ax2, ax3, state, T_window);
    drawnow limitrate;
end

%% Slider callback
function slider_callback(fig, sliderSigma, sliderK, textSigma, textK)
    state = getappdata(fig,'state');

    new_sigma = get(sliderSigma,'Value');
    new_K = get(sliderK,'Value');

    set(textSigma,'String',sprintf('%.2f',new_sigma));
    set(textK,'String',sprintf('%.2f',new_K));

    sigma_changed = abs(new_sigma - state.last_sigma) > 1e-10;
    K_changed = abs(new_K - state.last_K) > 1e-10;

    if sigma_changed
        % New sigma means new frequency ensemble and clear all measurements
        state.sigma = new_sigma;
        state.K = new_K;

        state.need_new_omega = true;

        state.live_K = new_K;
        state.live_r = NaN;

        state.last_sigma = new_sigma;
        state.last_K = new_K;

    elseif K_changed
        state.K = new_K;

        state.live_K = new_K;
        state.live_r = NaN;

        state.need_new_omega = false;
        state.last_K = new_K;

    else
        return;
    end

    state.need_reset = true;

    setappdata(fig,'state',state);
end

%% Initialize
function state = initialize_system(state)
    N = state.N;

    if state.need_new_omega || isempty(state.omega)
        omega = 1.0 + state.sigma * randn(N,1);
        omega = sort(omega,'ascend');
        state.omega = omega;
        state.need_new_omega = false;
    end

    state.theta = ones(N,1) * pi/2;

    state.t = 0;
    state.step = 0;

    state.r_history = [];
    state.spike_t = [];
    state.spike_i = [];
end

%% One step
function state = step_system(state, dt, record_window)
    theta = state.theta;
    omega = state.omega;
    K = state.K;

    z = mean(exp(1i * theta));
    r = abs(z);
    psi = angle(z);

    dtheta = omega + K * r * sin(psi - theta);
    theta_new = theta + dt * dtheta;

    crossed = top_crossing(theta, theta_new);
    idx = find(crossed);

    if ~isempty(idx)
        state.spike_t = [state.spike_t; state.t * ones(numel(idx),1)];
        state.spike_i = [state.spike_i; idx(:)];
    end

    state.theta = theta_new;
    state.t = state.t + dt;
    state.step = state.step + 1;

    state.r_history = [state.r_history; state.t, r];

    recent = state.r_history(state.r_history(:,1) > state.t - record_window, 2);

    if ~isempty(recent)
        state.live_r = mean(recent);
    else
        state.live_r = r;
    end

    keep = state.spike_t >= state.t - 150;
    state.spike_t = state.spike_t(keep);
    state.spike_i = state.spike_i(keep);
end

%% Top crossing
function crossed = top_crossing(theta_old, theta_new)
    top = pi/2;

    phi_old = wrapToPi_local(theta_old - top);
    phi_new = wrapToPi_local(theta_new - top);

    forward_cross = phi_old < 0 & phi_new >= 0 & abs(phi_new - phi_old) < pi;
    backward_cross = phi_old > 0 & phi_new <= 0 & abs(phi_new - phi_old) < pi;

    crossed = forward_cross | backward_cross;
end

%% Wrap
function y = wrapToPi_local(x)
    y = mod(x + pi, 2*pi) - pi;
end

%% Draw
function draw_all(ax1, ax2, ax3, state, T_window)
    draw_phase_circle(ax1, state);
    draw_transition_curve(ax2, state);
    draw_raster(ax3, state, T_window);
end

%% Panel 1
function draw_phase_circle(ax, state)
    cla(ax,'reset');

    theta = state.theta;
    N = state.N;

    a = linspace(0,2*pi,400);
    plot(ax,cos(a),sin(a),'k-','LineWidth',1.5);
    hold(ax,'on');

    plot(ax,[0 0],[0 1.15],'k--','LineWidth',1.2);

    colors = web_colormap(N);
    scatter(ax,cos(theta),sin(theta),52,colors,'filled', ...
        'MarkerEdgeColor','k','LineWidth',0.35);

    z = mean(exp(1i * theta));
    r = abs(z);
    psi = angle(z);

    quiver(ax,0,0,r*cos(psi),r*sin(psi),0, ...
        'k','LineWidth',2.4,'MaxHeadSize',0.5);

    axis(ax,'equal');
    xlim(ax,[-1.2 1.2]);
    ylim(ax,[-1.2 1.2]);

    xlabel(ax,'x');
    ylabel(ax,'y');
    title(ax,'Phase circle');

    box(ax,'on');
    set(ax,'LineWidth',1.2,'FontName','Arial');
end

%% Panel 2
function draw_transition_curve(ax, state)
    cla(ax,'reset');

    sigma = state.sigma;
    K_now = state.K;

    Kmax = 2.0;
    K_list = linspace(0,Kmax,500);
    r_theory = mean_field_r_curve(K_list, sigma);

    hMF = plot(ax,K_list,r_theory,'Color',[0.45 0.45 0.45], ...
        'LineWidth',2.4);
    hold(ax,'on');

    xline(ax,K_now,'k--','LineWidth',1.4, ...
        'HandleVisibility','off');

    hMeasLegend = scatter(ax,NaN,NaN,80, ...
        web_colormap(1),'filled','MarkerEdgeColor','k');

    if ~isnan(state.live_r)
        scatter(ax,state.live_K,state.live_r,95, ...
            web_colormap(1),'filled','MarkerEdgeColor','k','LineWidth',0.6, ...
            'HandleVisibility','off');
    end

    legend(ax,[hMF,hMeasLegend], ...
        {'MF prediction','Simulation'}, ...
        'Location','southeast', ...
        'Box','off');

    xlabel(ax,'K');
    ylabel(ax,'<r>');
    title(ax,'Phase diagram');

    xlim(ax,[0 Kmax]);
    ylim(ax,[0 1.02]);

    grid(ax,'off');
    box(ax,'off');
    set(ax,'LineWidth',1.2,'FontName','Arial','TickDir','out');
end

%% Panel 3
function draw_raster(ax, state, T_window)
    cla(ax,'reset');

    t0 = max(0,state.t - T_window);
    keep = state.spike_t >= t0;

    spike_t = state.spike_t(keep);
    spike_i = state.spike_i(keep);

    if ~isempty(spike_t)
        colors = web_colormap(state.N);
        scatter(ax,spike_t,spike_i,12,colors(spike_i,:),'filled');
    end

    xlim(ax,[t0 t0 + T_window]);
    ylim(ax,[0.5 state.N + 0.5]);

    xlabel(ax,'t');
    ylabel(ax,'i');
    title(ax,'Spike train');

    grid(ax,'off');
    box(ax,'off');
    set(ax,'LineWidth',1.2,'FontName','Arial','TickDir','out');
end

%% MF curve
function r_curve = mean_field_r_curve(K_list, sigma)
    r_curve = zeros(size(K_list));

    if sigma <= 0
        r_curve(K_list > 0) = 1;
        return;
    end

    Kc = 2 * sqrt(2/pi) * sigma;

    for kk = 1:length(K_list)
        K = K_list(kk);

        if K <= Kc
            r_curve(kk) = 0;
        else
            fun = @(r) self_consistency_residual(r, K, sigma);

            try
                r_curve(kk) = fzero(fun,[1e-6,0.999999]);
            catch
                r_grid = linspace(1e-4,0.999,300);
                vals = arrayfun(@(r) abs(fun(r)), r_grid);
                [~,id] = min(vals);
                r_curve(kk) = r_grid(id);
            end
        end
    end
end

%% Self-consistency
function F = self_consistency_residual(r, K, sigma)
    x = linspace(-1,1,2001);

    g = @(w) 1/(sqrt(2*pi)*sigma) * exp(-w.^2/(2*sigma^2));

    integrand = sqrt(1 - x.^2) .* g(K*r*x);

    rhs = K * r * trapz(x, integrand);

    F = r - rhs;
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
