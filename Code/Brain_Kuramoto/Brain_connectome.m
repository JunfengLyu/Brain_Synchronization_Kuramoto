%% Brain Connectome Construction
%% Construct group structural connectome and identify hub regions

clear; clc; close all;

dataDir = 'data';

raw = load(fullfile(dataDir,'rawdata.mat'));

SC = raw.SCmatrices;

[S,N,~] = size(SC);

%% Subject-wise preprocessing

processed = zeros(size(SC));

for i = 1:S

    M = squeeze(SC(i,:,:));

    M = (M + M')/2;

    M = M - diag(diag(M));

    processed(i,:,:) = M;

end

%% Group connectome

prob = squeeze(mean(processed > 0,1));

M_group = double(prob >= 0.40);

M_group = M_group - diag(diag(M_group));

%% Degree and hub detection

degree = sum(M_group,2);

cutoff = prctile(degree,85);

hubs = degree >= cutoff;

fprintf('Number of regions = %d\n',N);
fprintf('Hub threshold (85th percentile) = %.1f\n',cutoff);
fprintf('Number of hubs = %d\n',sum(hubs));

%% Visualization

fig = figure( ...
    'Color','w', ...
    'Position',[100 100 725 325]);

%% Panel A: Group connectome

axA = subplot(1,2,1);

imagesc(M_group);

axis image;

colormap(axA,gray);

cb = colorbar;
cb.Label.String = 'Edge exists';

xlabel('Brain region index');
ylabel('Brain region index');

set(gca,...
    'XTick',[1 30 60 90],...
    'YTick',[1 30 60 90],...
    'FontSize',10,...
    'LineWidth',1);

text(-0.16,1.04,'A',...
    'Units','normalized',...
    'FontSize',12);

%% Panel B: Degree distribution

axB = subplot(1,2,2);

hold on;

bar(1:N,...
    degree,...
    'FaceColor',[0.85 0.87 0.89],...
    'EdgeColor','none',...
    'DisplayName','Non-hub');

bar(find(hubs),...
    degree(hubs),...
    'FaceColor',[0.82 0.26 0.36],...
    'EdgeColor','none',...
    'DisplayName','Hub');

yline(cutoff,...
    '--',...
    'Color',[0.05 0.05 0.05],...
    'LineWidth',1.3,...
    'DisplayName',sprintf('85th percentile = %.1f',cutoff));

xlabel('Brain region index');
ylabel('Node degree');

xlim([0 N+1]);
ylim([0 max(degree)*1.18]);

legend(...
    'Location','northoutside',...
    'Orientation','horizontal',...
    'Box','off');

set(gca,...
    'FontSize',10,...
    'LineWidth',1,...
    'TickDir','out');

text(-0.12,1.04,'B',...
    'Units','normalized',...
    'FontSize',12);

%% Layout

set(axA,'Position',[0.08 0.20 0.32 0.64]);
set(axB,'Position',[0.56 0.20 0.39 0.64]);

%% Optional save

% exportgraphics(fig,...
%     '06_connectome_construction.png',...
%     'Resolution',300,...
%     'BackgroundColor','w');