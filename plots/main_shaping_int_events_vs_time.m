close all
clear
clc

%% Read Data
load('shaping_int_events_vs_time.mat')

%% Main
mean_distance = zeros(5,100);
max_distance = zeros(5,100);
min_distance = inf(5,100);
for itr=1:20
    for i=1:tf
        for method=1:5
            distance = norm(count_user_events(times{method,itr}, users{method,itr}, n, 0, i) - ell);
            mean_distance(method,i) = mean_distance(method,i) + distance;
            if distance > max_distance(method,i)
                max_distance(method,i) = distance;
            end
            if distance < min_distance(method,i)
                min_distance(method,i) = distance;
            end
        end
    end
end
for method=1:5
    mean_distance(method,:) = mean_distance(method,:)/20;
end
    
scale_factor = 1.5;

%% Plots
c1 = [0         0.4470    0.7410];
c2 = [0.8500    0.3250    0.0980];
c3 = [0.9290    0.6940    0.1250];
c4 = [0.4940    0.1840    0.5560];
c5 = [0.4660    0.6740    0.1880];
% 0.3010    0.7450    0.9330
% 0.6350    0.0780    0.1840

set(0,'defaulttextInterpreter','none')
% --------------------------------------------------------------------------
figure(1);
plot(1:tf, mean_distance(1,:),'MarkerSize',3,'linewidth',0.9);
% jbfill(1:tf, max_distance(1,:), min_distance(1,:), c1, c1, 1, 0.5)
hold on
plot(1:tf, mean_distance(2,:),'MarkerSize',3,'linewidth',0.9);
% jbfill(1:tf, max_distance(2,:), min_distance(2,:), c2, c2, 1, 0.5)
hold on
plot(1:tf, mean_distance(3,:),'MarkerSize',3,'linewidth',0.9);
% jbfill(1:tf, max_distance(3,:), min_distance(3,:), c3, c3, 1, 0.5)
hold on
plot(1:tf, mean_distance(4,:),'MarkerSize',3,'linewidth',0.9);
% jbfill(1:tf, max_distance(4,:), min_distance(4,:), c4, c4, 1, 0.5)
hold on
plot(1:tf, mean_distance(5,:),'MarkerSize',3,'linewidth',0.9);
% jbfill(1:tf, max_distance(5,:), min_distance(5,:), c5, c5, 1, 0.5)

% % xlim([0, 225])
% % ylim([0, 82])
xlabel('time')
ylabel('$\ell_2$ distance')
% % 
% % text(1,31000,'$\scriptstyle{\times 10^4}$')
% lh = legend('OPT','UNF','DEG','PRK','UNC','Location', 'northwest');
% set(lh,'Interpreter','latex')
% % set(gca,'xtick',[0 200 400 600 800 1000])
% % 
grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) 500 400]);
laprint(1,'shaping_int_events_vs_time','factor',scale_factor)%,'asonscreen','on')

% --------------------------------------------------------------------------
% figure(2);
% plot(times{1,itr},'MarkerSize',3,'linewidth',0.9);
% hold on
% plot(times{2,itr},'MarkerSize',3,'linewidth',0.9);
% hold on
% plot(times{3,itr},'MarkerSize',3,'linewidth',0.9);
% hold on
% plot(times{4,itr},'MarkerSize',3,'linewidth',0.9);
% hold on
% plot(times{5,itr},'MarkerSize',3,'linewidth',0.9);
% lh = legend('OPT','UNF','DEG','PRK','UNC','Location', 'northwest');
% grid on
% pf = get(2,'position');
% set(2,'position',[pf(2) pf(2) 500 400]);
% laprint(2,'shaping_int_events_vs_budget','factor',scale_factor)%,'asonscreen','on')

%% Functions
function count = count_user_events(times, users, n, a, b)
    users = users + 1;
    count = zeros(1,n);
    for i = 1:length(times)
        if times(i) < b && times(i) > a
            count(users(i)) = count(users(i)) + 1;
        end
    end
end
