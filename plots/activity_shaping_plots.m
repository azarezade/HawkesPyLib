close all
clear
clc

%% Terminal Objective vs Budget

%% Terminal EventsNum vs Budget

%% Integral Objective vs Budget

%% Integral EventsNum vs Budget
% load data
load('../results/shaping_int_events_vs_budget.mat')

% plot params 
scale_factor = 1.5;
set(0,'defaulttextInterpreter','none')

figure(1);
plot(budget, obj(1,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(2,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(3,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(4,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
% xlim([0, 225])
% ylim([0, 82])
xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\|\overline{\int dN(s) ds} - \ell \|_2^2$')
% text(1,31000,'$\scriptstyle{\times 10^4}$')
lh = legend('DEG','PRK','UNF', 'OPT', 'Location', 'northwest');
set(lh,'Interpreter','latex')
% set(gca,'xtick',[0 200 400 600 800 1000])
grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) 500 400]);
laprint(1,'shaping_int_events_vs_budget','factor',scale_factor) % 'asonscreen','on

