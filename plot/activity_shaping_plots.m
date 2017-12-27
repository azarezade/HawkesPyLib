close all
clear
clc

%% Terminal Objective vs Budget

%% Terminal EventsNum vs Budget
filename = 'shaping_events_vs_budget';
load(fullfile('..','result',strcat(filename,'.mat')));

% plot data
xlabel = '$c$ (bubget)';
ylabel = '\vspace{-0.5mm} $\|\overline{dN(T)} - \ell \|_2^2$'; 
legend = {'DEG','PRK','UNF', 'OPT', 'UNC', 'northwest'};
texplot(filename, budget, obj, xlabel, ylabel, legend)

% %% Integral Objective vs Budget
% filename = 'shaping_int_obj_vs_budget';
% load(fullfile('..','result',strcat(filename,'.mat')));
% 
% % plot data
% xlabel = '$c$ (bubget)';
% ylabel = '\vspace{-0.5mm} $\|\mathbb{E}[\int_0^T dN(s) \, ds] - \ell \|_2^2$'; 
% legend = {'DEG','PRK','UNF', 'OPT', 'northwest'};
% texplot(filename, budget, obj([1,3,5,7],:), xlabel, ylabel, legend)
% 
% %% Integral EventsNum vs Budget
% % load file
% filename = 'shaping_int_events_vs_budget';
% load(fullfile('..','result',strcat(filename,'.mat')));
% 
% % plot data
% xlabel = '$c$ (bubget)';
% ylabel = '\vspace{-0.5mm} $\|\overline{\int dN(s) ds} - \ell \|_2^2$'; 
% legend = {'DEG','PRK','UNF', 'OPT', 'northwest'};
% texplot(filename, budget, obj(1:end-1,:), xlabel, ylabel, legend)
% 
% %% Finalize
% close all