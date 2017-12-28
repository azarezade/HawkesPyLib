function [  ] = texprint( filename )
% Write latex file and compile it to generate the pdf of the plot

%----- Write Latex File -----
fid = fopen('latex_plot.tex','w');
fprintf(fid, ['\\documentclass[10pt,border=8pt]{standalone} \n',...
              '\\usepackage{graphicx,color,psfrag} \n',...
              '\\usepackage{amsmath,amsfonts,amssymb,bm} \n',...
              '\\begin{document} \n',...
              '\\input{%s} \n',...
              '\\end{document} \n'], filename);

%----- Compile Latex File -----
% set the path for latex, dvips, ps2pdf which are not in /bin/bash path
latex = '/Library/TeX/texbin/latex ';
dvips = '/Library/TeX/texbin/dvips';
ps2pdf = '/usr/local/bin/ps2pdf';

% run shell command to complie latex file 
compile = sprintf('%s latex_plot.tex; %s latex_plot.dvi; %s latex_plot.ps %s.pdf;',latex, dvips, ps2pdf, filename);
clean = sprintf('rm *.{dvi,log,pfg,aux,ps,eps,tex}');
system(compile);
system(clean);

end

