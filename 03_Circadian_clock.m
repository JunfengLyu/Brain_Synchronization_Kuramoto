clear; clc; close all;
projectRoot = fileparts(mfilename('fullpath'));
runSection(projectRoot, fullfile('Code','06_circadian_resynchronization','matlab','circadian_recovery.m'));
fprintf('Generated circadian-clock figures 15-17 in Report/Figs.\n');

function runSection(projectRoot, relativeScript)
oldDir = pwd;
scriptPath = fullfile(projectRoot, relativeScript);
cleanup = onCleanup(@() cd(oldDir));
cd(fileparts(scriptPath));
run(scriptPath);
end
