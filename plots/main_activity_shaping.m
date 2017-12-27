close all
clear
clc

%% Read Data
load('shaping_obj_vs_budget.mat')
obj_mehrdad = [41.0602
   36.8833
   32.6496
   28.8448
   25.4690
   16.2545
   14.2205
   11.8728
   11.5925
   11.5925
   11.5925];

idx = logical([1 0 1 0 1 0  1 1 1 1 1]);
budget = budget(idx);
obj = obj(:,idx);
obj_mehrdad = obj_mehrdad(idx);
   
scale_factor = 1.5;
%% Plots
set(0,'defaulttextInterpreter','none') 
% --------------------------------------------------------------------------
figure(1);
plot(budget, obj(1,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
% hold on
% plot(budget, obj(2,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(3,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
% hold on
% plot(budget, obj(4,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(5,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
% hold on
% plot(budget, obj(6,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj_mehrdad,'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(7,:),'Marker','o','MarkerSize',3,'linewidth',0.9);

% xlim([0, 225])
ylim([0, 82])
xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\|\mathbb{E}[dN(T)] - \ell \|_2^2$')
% 
% text(1,31000,'$\scriptstyle{\times 10^4}$')
lh = legend('DEG','PRK','UNF', 'OPL','OPT', 'Location', 'northwest');
set(lh,'Interpreter','latex')
% set(gca,'xtick',[0 200 400 600 800 1000])
% 
grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) 500 400]);
laprint(1,'shaping_obj_vs_budget','factor',scale_factor)%,'asonscreen','on')
