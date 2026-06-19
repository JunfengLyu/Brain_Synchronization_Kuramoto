%% Network Topology Comparison
%% Synchronization transition on global, ER and BA networks

clear;clc;close all;

%% Parameters

N=100;
t_end=200;
dt=0.01;
tspan=0:dt:t_end;

sigma=0.25;

rng(42);

omega=1+sigma*randn(N,1);

K_range=0:0.2:8;
nK=length(K_range);

%% Generate Networks

A_global=ones(N)-eye(N);

p_ER=0.10;

A_ER=rand(N)<p_ER;
A_ER=triu(A_ER,1);
A_ER=A_ER+A_ER';
A_ER(1:N+1:end)=0;

m0=5;
m=5;

A_BA=BA_network(N,m0,m);

fprintf('Average degree:\n');
fprintf('  Global: %.1f\n',mean(sum(A_global,2)));
fprintf('  ER:     %.1f\n',mean(sum(A_ER,2)));
fprintf('  BA:     %.1f\n',mean(sum(A_BA,2)));

fprintf('Max degree:\n');
fprintf('  Global: %.1f\n',max(sum(A_global,2)));
fprintf('  ER:     %.1f\n',max(sum(A_ER,2)));
fprintf('  BA:     %.1f\n',max(sum(A_BA,2)));

%% Simulate Synchronization Transition

R_global=zeros(1,nK);
R_ER=zeros(1,nK);
R_BA=zeros(1,nK);

fprintf('\nStarting simulation over %d K values...\n',nK);

tic;

for idx=1:nK

    K=K_range(idx);

    if mod(idx,10)==0
        fprintf('Progress %.1f%%   K=%.2f\n',idx/nK*100,K);
    end

    [~,theta]=ode45( ...
        @(t,theta)kuramoto_ode_standard(t,theta,omega,K,A_global), ...
        tspan, ...
        2*pi*rand(N,1));

    R_global(idx)=compute_order_parameter(theta(end-1000:end,:));

    [~,theta]=ode45( ...
        @(t,theta)kuramoto_ode_standard(t,theta,omega,K,A_ER), ...
        tspan, ...
        2*pi*rand(N,1));

    R_ER(idx)=compute_order_parameter(theta(end-1000:end,:));

    [~,theta]=ode45( ...
        @(t,theta)kuramoto_ode_standard(t,theta,omega,K,A_BA), ...
        tspan, ...
        2*pi*rand(N,1));

    R_BA(idx)=compute_order_parameter(theta(end-1000:end,:));

end

toc;

%% Critical Coupling

threshold=0.2;

idx_global=find(R_global>threshold,1,'first');
idx_ER=find(R_ER>threshold,1,'first');
idx_BA=find(R_BA>threshold,1,'first');

Kc_global=K_range(idx_global);
Kc_ER=K_range(idx_ER);
Kc_BA=K_range(idx_BA);

fprintf('\nCritical coupling:\n');
fprintf('Global : %.2f\n',Kc_global);
fprintf('ER     : %.2f\n',Kc_ER);
fprintf('BA     : %.2f\n',Kc_BA);

%% Visualization

C=plasma(256);

color_global=C(40,:);
color_ER=C(140,:);
color_BA=C(220,:);

figure('Color','w','Position',[100 100 545 325]);
hold on;

plot(K_range,R_global,'-o',...
    'Color',color_global,...
    'LineWidth',1.8,...
    'MarkerSize',3.4,...
    'MarkerFaceColor',color_global,...
    'DisplayName','Global');

plot(K_range,R_ER,'-s',...
    'Color',color_ER,...
    'LineWidth',1.8,...
    'MarkerSize',3.4,...
    'MarkerFaceColor',color_ER,...
    'DisplayName','ER random');

plot(K_range,R_BA,'-^',...
    'Color',color_BA,...
    'LineWidth',1.8,...
    'MarkerSize',3.4,...
    'MarkerFaceColor',color_BA,...
    'DisplayName','BA scale-free');

xline(Kc_global,'--',...
    'Color',color_global,...
    'LineWidth',1.2,...
    'HandleVisibility','off');

xline(Kc_ER,'--',...
    'Color',color_ER,...
    'LineWidth',1.2,...
    'HandleVisibility','off');

xline(Kc_BA,'--',...
    'Color',color_BA,...
    'LineWidth',1.2,...
    'HandleVisibility','off');

text(Kc_global+0.08,0.50,...
    sprintf('$K_c=%.2f$',Kc_global),...
    'Interpreter','latex',...
    'Color',color_global,...
    'FontSize',8);

text(Kc_BA+0.08,0.42,...
    sprintf('$K_c=%.2f$',Kc_BA),...
    'Interpreter','latex',...
    'Color',color_BA,...
    'FontSize',8);

text(Kc_ER+0.08,0.58,...
    sprintf('$K_c=%.2f$',Kc_ER),...
    'Interpreter','latex',...
    'Color',color_ER,...
    'FontSize',8);

xlabel('Coupling strength K');
ylabel('Order parameter R');

xlim([0 8]);
ylim([0 1.04]);

lgd=legend('Location','eastoutside');
lgd.Box='off';

set(gca,...
    'FontName','Arial',...
    'FontSize',10,...
    'LineWidth',1.0,...
    'TickDir','out');

box off;
exportgraphics(gcf,...
    '05_network_topology_transition.png',...
    'Resolution',300,...
    'BackgroundColor','w');

%% Helper Functions

function dtheta=kuramoto_ode_standard(~,theta,omega,K,A)

N=length(theta);

dtheta=zeros(N,1);

for i=1:N

    neighbors=find(A(i,:));

    if ~isempty(neighbors)

        interaction=sum(sin(theta(neighbors)-theta(i)));

        dtheta(i)=omega(i)+(K/N)*interaction;

    else

        dtheta(i)=omega(i);

    end

end

end

function R=compute_order_parameter(theta_trajectory)

n_steps=size(theta_trajectory,1);

R_step=zeros(n_steps,1);

for t=1:n_steps

    z=mean(exp(1i*theta_trajectory(t,:)));

    R_step(t)=abs(z);

end

R=mean(R_step);

end

function A=BA_network(N,m0,m)

A=zeros(N);

A(1:m0,1:m0)=1-eye(m0);

degree=sum(A);

for i=m0+1:N

    total_deg=sum(degree(1:i-1));

    prob=degree(1:i-1)/total_deg;

    selected=[];

    while length(selected)<m

        r=rand();

        cumsum_prob=cumsum(prob);

        candidate=find(cumsum_prob>=r,1,'first');

        if ~ismember(candidate,selected)
            selected=[selected,candidate];
        end

    end

    A(i,selected)=1;
    A(selected,i)=1;

    degree(i)=m;
    degree(selected)=degree(selected)+1;

end

end

function cmap=plasma(m)

if nargin<1,m=256;end

base=[...
0.0504 0.0298 0.5280;
0.2900 0.0700 0.6500;
0.5100 0.1500 0.6500;
0.7100 0.2800 0.5100;
0.8700 0.4700 0.3200;
0.9800 0.7400 0.1700;
0.9400 0.9750 0.1310];

x=linspace(0,1,size(base,1));
xi=linspace(0,1,m);

cmap=interp1(x,base,xi,'pchip');

end