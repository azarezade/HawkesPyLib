close all
clear
clc

%% Maximization Terminal Objective vs Budget

%% Maximization Terminal EventsNum vs Budget

%% Maximization Integral Objective vs Budget

%% Maximization Integral EventsNum vs Budget

%% Shaping Terminal Objective vs Budget
filename = 'shaping_obj_vs_budget';
load(fullfile('..','result',strcat(filename,'.mat')));

% add Mehrdad results to obj matrix
obj_mehrdad = [41.0602, 32.6496, 25.4690, 14.2205, 11.8728, 11.5925, 11.5925, 11.5925];
obj = [obj; obj(end,:)];
obj(end-1,:) = obj_mehrdad;

% plot data
xlabel = '$c$ (bubget)';
ylabel = '\vspace{-0.5mm} $\|\mathbb{E}[dN(T)] - \ell \|_2^2$'; 
legend = {'DEG','PRK','UNF', 'OPL', 'OPT', 'northwest'};
texplot(filename, budget, obj, xlabel, ylabel, legend)

%% Shaping Terminal EventsNum vs Budget
filename = 'shaping_events_vs_budget';
load(fullfile('..','result',strcat(filename,'.mat')));

% plot data
xlabel = '$c$ (bubget)';
ylabel = '\vspace{-0.5mm} $\|\overline{dN(T)} - \ell \|_2^2$'; 
legend = {'DEG','PRK','UNF', 'OPT', 'UNC', 'northwest'};
texplot(filename, budget, obj, xlabel, ylabel, legend)

%% Shaping Integral Objective vs Budget
filename = 'shaping_int_obj_vs_budget';
load(fullfile('..','result',strcat(filename,'.mat')));

% plot data
xlabel = '$c$ (bubget)';
ylabel = '\vspace{-0.5mm} $\|\mathbb{E}[\int_0^T dN(s) \, ds] - \ell \|_2^2$'; 
legend = {'DEG','PRK','UNF', 'OPT', 'northwest'};
texplot(filename, budget, obj([1,3,5,7],:), xlabel, ylabel, legend)

%% Shaping Integral EventsNum vs Budget
% load file
filename = 'shaping_int_events_vs_budget';
load(fullfile('..','result',strcat(filename,'.mat')));

% plot data
xlabel = '$c$ (bubget)';
ylabel = '\vspace{-0.5mm} $\|\overline{\int dN(s) ds} - \ell \|_2^2$'; 
legend = {'DEG','PRK','UNF', 'OPT', 'northwest'};
texplot(filename, budget, obj(1:end-1,:), xlabel, ylabel, legend)

%% Finalize
close all