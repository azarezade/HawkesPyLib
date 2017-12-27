close all
clear
clc

%% Read Data
load('shaping_events_vs_budget.mat')

% idx = logical([1 0 1 0 1 0  1 1 1 1 1]);
% budget = budget(idx);
% obj = obj(:,idx);

sz = size(event_num);
obj = zeros(5,8);
for i=1:sz(1)
    for j=1:sz(2)
        obj(i,j) = norm(squeeze(event_num(i,j,:)) - ell')^2;
    end
end
    
scale_factor = 1.5;
%% Plots
set(0,'defaulttextInterpreter','none') 
% --------------------------------------------------------------------------
figure(1);
plot(budget, obj(1,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(2,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(3,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(4,:),'Marker','o','MarkerSize',3,'linewidth',0.9);
hold on
plot(budget, obj(5,:),'Marker','o','MarkerSize',3,'linewidth',0.9);

% xlim([0, 225])
% ylim([0, 82])
xlabel('$c$ (bubget)')
ylabel('\vspace{-0.5mm} $\|\overline{dN(T)} - \ell \|_2^2$')
% 
% text(1,31000,'$\scriptstyle{\times 10^4}$')
lh = legend('DEG','PRK','UNF', 'OPT','UNC', 'Location', 'northwest');
set(lh,'Interpreter','latex')
% set(gca,'xtick',[0 200 400 600 800 1000])
% 
grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) 500 400]);
laprint(1,'shaping_events_vs_budget','factor',scale_factor)%,'asonscreen','on')
