function [ ] = texplot( filename, x, y, xlbl, ylbl, lgnd, varargin )
% plot y vs x with given params and output a pdf file

% check number of optinal args
numvarargs = length(varargin);
if numvarargs > 3
    error('texplot:TooManyInputs', 'requires at most 3 optional inputs');
end

% set defaults for optional args
optargs = {{0,0,''}, 500, 400, 'o', 3, 0.9, 1.5};

% overwrite optinal args if are given in inputs
optargs(1:numvarargs) = varargin;

% set name to optional args
[txt, width, height, mtype, msize, lwidth, sfactor] = optargs{:};

% plot
set(0,'defaulttextInterpreter','none')
figure(1);
for i=1:size(y,1)
    plot(x, y(i,:),'Marker',mtype,'MarkerSize',msize,'linewidth',lwidth);
    hold on
end
hold off

% plot settings
xlabel(xlbl)
ylabel(ylbl)
text(txt{:})
lh = legend(lgnd{(1:end-1)}, 'Location', lgnd{[end]});
set(lh,'Interpreter','latex')

grid on
pf = get(1,'position');
set(1,'position',[pf(2) pf(2) width height]);

laprint(1,filename,'factor',sfactor) % 'asonscreen','on
texprint(filename)

end

