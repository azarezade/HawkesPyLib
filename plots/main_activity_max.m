close all
clear
clc

%% Read Data
max_events_vs_budget = textread('max_eta_event_num_vs_budget_mean.txt');
max_events_vs_budget_mehrdad = textread('max_eta_event_num_vs_budget_mehrdad_mean.txt');

max_terminal_events_vs_budget = textread('max_eta_terminal_event_num_vs_budget_mean.txt');
max_terminal_events_vs_budget_mehrdad = textread('max_eta_terminal_event_num_vs_budget_mehrdad_mean.txt');

max_obj_vs_budget = textread('max_eta_obj_vs_budget.txt');
max_obj_vs_budget_mehrdad = textread('max_eta_obj_vs_budget_mehrdad.txt');

max_int_obj_vs_budget = textread('max_int_eta_obj_vs_budget.txt');
max_int_events_vs_budget = textread('max_int_eta_event_num_vs_budget_mean.txt');
max_int_terminal_events_vs_budget = textread('max_int_eta_terminal_event_num_vs_budget_mean.txt');

c = 1;
budget = [1*c, 100*c, 200*c, 300*c, 400*c, 500*c];
idx = logical([1 1 1 1 1 1]);

scale_factor = 1.5;
%% Plots
set(0,'defaulttextInterpreter','none') 
% --------------------------------------------------------------------------
figure(1);
plot(budget(idx), max_events_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_events_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_events_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_events_vs_budget_mehrdad(idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_events_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);


xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\sum_i \int_0^T dN_i(s)\,ds$')

text(1,31000,'$\scriptstyle{\times 10^4}$')
% lh = legend('DEG','PRK','UNF', 'OPL' ,'OPT', 'Location', 'northwest');
% set(lh,'Interpreter','latex')
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) 500 400]);
laprint(1,'max_events_vs_budget','factor',scale_factor)%,'asonscreen','on')

%--------------------------------------------------------------------------
figure(2);
plot(budget(idx), max_obj_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_obj_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_obj_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_obj_vs_budget_mehrdad(idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_obj_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);

xlabel('$c$ (bubget)')
ylabel('$\bm{w}^T \mathbb{E}[d\bm{N}(T)]$')
% ylabel('$\sum_i \eta_i(T)$')

text(1, 5.2e4, '$\scriptstyle{\times 10^4}$')
lh = legend('DEG','PRK','UNF', 'OPL' ,'OPT', 'Location', 'northwest');
set(lh,'Interpreter','latex')
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(2,'position');
set(2,'position',[pf(2) pf(2) 500 400]);
laprint(2,'max_obj_vs_budget','factor',scale_factor)%,'asonscreen','on')

%--------------------------------------------------------------------------
figure(3);
plot(budget(idx), max_terminal_events_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_terminal_events_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_terminal_events_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_terminal_events_vs_budget_mehrdad(idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_terminal_events_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);

xlabel('$c$ (bubget)')
ylabel('$\sum_i N_i(T)$')

% text(3,31000,'$\scriptstyle{\times 10^4}$')
% lh=legend('DEG','PRK','UNF', 'OPT', 'Location', 'northwest');
% set(lh,'Interpreter','latex')
% ylim([5, 1000])
% set(gca,'ytick',[0 1 2 3]*1e4)
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(3,'position');
set(3,'position',[pf(2) pf(2) 500 400]);
laprint(3,'max_terminal_events_vs_budget','factor',scale_factor)%,'asonscreen','on')

%--------------------------------------------------------------------------
figure(4);
plot(budget(idx), max_int_events_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_events_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_events_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_events_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);

xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\sum_i \int_0^T dN_i(s)\,ds$')

text(3,21000,'$\scriptstyle{\times 10^3}$')
% lh=legend('DEG','PRK','UNF', 'OPT', 'Location', 'northwest');
% set(lh,'Interpreter','latex')
% ylim([5, 1000])
% set(gca,'ytick',[0 1 2 3]*1e4)
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(4,'position');
set(4,'position',[pf(2) pf(2) 500 400]);
laprint(4,'max_int_events_vs_budget','factor',scale_factor)%,'asonscreen','on')

%--------------------------------------------------------------------------
figure(5);
plot(budget(idx), max_int_obj_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_obj_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_obj_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_obj_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);

xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\bm{w}^T\, \mathbb{E}[ \int_{t_0}^{t_f} d\bm{N}(s)]$')

text(3,21000,'$\scriptstyle{\times 10^3}$')
lh=legend('DEG','PRK','UNF', 'OPT', 'Location', 'northwest');
set(lh,'Interpreter','latex')
% ylim([5, 1000])
% set(gca,'ytick',[0 1 2 3]*1e4)
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(5,'position');
set(5,'position',[pf(2) pf(2) 500 400]);
laprint(5,'max_int_obj_vs_budget','factor',scale_factor)%,'asonscreen','on')

%--------------------------------------------------------------------------
figure(6);
plot(budget(idx), max_int_terminal_events_vs_budget(1,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_terminal_events_vs_budget(2,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_terminal_events_vs_budget(3,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget(idx), max_int_terminal_events_vs_budget(4,idx),'Marker','o','MarkerSize',3,'linewidth',0.9);

xlabel('$c$ (bubget)')
ylabel('$\sum_i N_i(T)$')

% text(3,21000,'$\scriptstyle{\times 10^3}$')
% lh=legend('DEG','PRK','UNF', 'OPT', 'Location', 'northwest');
% set(lh,'Interpreter','latex')
% ylim([5, 1000])
% set(gca,'ytick',[0 1 2 3]*1e4)
% set(gca,'xtick',[0 200 400 600 800 1000])

grid on
pf = get(6,'position');
set(6,'position',[pf(2) pf(2) 500 400]);
laprint(6,'max_int_terminal_events_vs_budget','factor',scale_factor)%,'asonscreen','on')

