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

fig = figure;
hold on
contourf(ell, sn, log(nlZ), 50, DisplayName='Log L')
scatter(ell(minima), sn(minima), '+r', DisplayName='Local Minima')
xlabel('ln(\lambda)', Interpreter='tex')
ylabel('ln(\sigma_n)', Interpreter='tex')
c = colorbar;
c.Label.String = 'ln(L)';
legend
fig.Position = [0,0,800,420];


saveas(fig,'figures/B/model_evidence_contour','epsc')

% Compare local minima
hyp_init = struct();
hyp_init.mean = [];
hyp_init.cov = [-5 0];
hyp_init.lik = -2;

hyp_init_2.mean = [];
hyp_init_2.cov = [5 0];
hyp_init_2.lik = -0.3;

hyp_init(2) = hyp_init_2;


xs = linspace(-3, 3, 1001)';
for i = 1:length(hyp_init)
    hyp_min = minimize(hyp_init(i), @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    [mu, s2] = gp(hyp_min, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    nlZ_min = gp(hyp_min, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    
    fig = figure;
    hold on
    fill([xs; flipdim(xs,1)], f, [7 7 7]/8, DisplayName='95% Prediction Error Bars')
    plot(xs, mu, DisplayName='Prediction Mean'); 
    scatter(x, y, '+', DisplayName='Data');
    ylabel('Output - y')
    fprintf('ell = %f, sf = %f, sn = %f -> nlZ = %f\n', exp(hyp_min.cov(1)), exp(hyp_min.cov(2)), exp(hyp_min.lik(1)), nlZ_min)
    xlabel('Input - x')
    legend

    saveas(fig,sprintf('figures/B/hp_optimum_comparison_%d', i),'epsc')
end