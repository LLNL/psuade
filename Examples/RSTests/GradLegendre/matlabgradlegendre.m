if exist('noCLF') 
   hold off
else
   clf
end;
A = [
3.333333e-01
3.333333e-01
3.333333e-01
];
bar(A, 0.8);
set(gca,'linewidth',2)
set(gca,'fontweight','bold')
set(gca,'fontsize',12)
grid on
box on
title('Legendre VCE Rankings','FontWeight','bold','FontSize',12)
xlabel('Input parameters','FontWeight','bold','FontSize',12)
ylabel('Rank Metric','FontWeight','bold','FontSize',12)
