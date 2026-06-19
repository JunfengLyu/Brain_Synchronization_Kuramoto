clear; clc; close all;
projectRoot = fileparts(mfilename('fullpath'));
runSection(projectRoot, fullfile('Code','03_connection_architecture','matlab','topology_connectome.m'));
runSection(projectRoot, fullfile('Code','04_brain_network_synchronization','matlab','brain_sync.m'));
runSection(projectRoot, fullfile('Code','05_alzheimers_disease','matlab','ad_continuum.m'));
fprintf('Generated brain connectome and AD figures 06-14 in Report/Figs.\n');

function runSection(projectRoot, relativeScript)
oldDir = pwd;
scriptPath = fullfile(projectRoot, relativeScript);
cleanup = onCleanup(@() cd(oldDir));
cd(fileparts(scriptPath));
run(scriptPath);
end
