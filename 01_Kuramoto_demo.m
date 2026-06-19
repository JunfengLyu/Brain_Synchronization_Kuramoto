clear; clc; close all;
projectRoot = fileparts(mfilename('fullpath'));
runSection(projectRoot, fullfile('Code','02_review_on_kuramoto_model','matlab','Kuramoto_demo.m'));
runSection(projectRoot, fullfile('Code','02_review_on_kuramoto_model','matlab','Kuramoto_transition.m'));
runSection(projectRoot, fullfile('Code','03_connection_architecture','matlab','topology_connectome.m'));
fprintf('Generated Kuramoto figures 02-05 in Report/Figs.\n');

function runSection(projectRoot, relativeScript)
oldDir = pwd;
scriptPath = fullfile(projectRoot, relativeScript);
cleanup = onCleanup(@() cd(oldDir));
cd(fileparts(scriptPath));
run(scriptPath);
end
