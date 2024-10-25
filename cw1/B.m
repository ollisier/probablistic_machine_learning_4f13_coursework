clear all
close all

load("cw1/data/cw1a.mat")

meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;


% Sweep hyper-parameters and plot slices 
sf = 0;
[ell, sn] = meshgrid(-4:0.1:4, -4:0.1:1);
nlZ = zeros(size(ell));
for i = 1:numel(ell)
    hyp = struct('mean', [], 'cov', [ell(i) sf], 'lik', sn(i));
    nlZ(i) = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
end
minima = islocalmin2(nlZ);

figure
hold on
contourf(ell, sn, log(nlZ), 50, DisplayName='Log Log Negative Marginal Likelihood')
scatter(ell(minima), sn(minima), '+r', DisplayName='Local Optima')
title(sprintf('\\sigma_s = %f', exp(sf)))
xlabel('ln(\lambda)', Interpreter='tex')
ylabel('ln(\sigma_n)', Interpreter='tex')
c = colorbar;
c.Label.String = 'ln(ln(-Z_{|y}))';
legend
saveas(gcf,'figures/B/model_evidence_contour','epsc')

% Compare local minima
hyp_init = struct();
hyp_init.mean = [];
hyp_init.cov = [-5 0];
hyp_init.lik = -2;

hyp_init_2.mean = [];
hyp_init_2.cov = [5 0];
hyp_init_2.lik = -0.3;

hyp_init(2) = hyp_init_2;


figure
xs = linspace(-3, 3, 1001)';
for i = 1:length(hyp_init)
    hyp_min = minimize(hyp_init(i), @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    [mu, s2] = gp(hyp_min, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    nlZ_min = gp(hyp_min, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    
    subplot(length(hyp_init),1,i)
    hold on
    fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
    plot(xs, mu); 
    scatter(x, y, '+');
    ylabel('Output - y')
    fprintf('ell = %f, sf = %f, sn = %f -> nlZ = %f\n', exp(hyp_min.cov(1)), exp(hyp_min.cov(2)), exp(hyp_min.lik(1)), -nlZ_min)
end
xlabel('Input - x')
saveas(gcf,'figures/B/hp_optimum_comparison','epsc')
