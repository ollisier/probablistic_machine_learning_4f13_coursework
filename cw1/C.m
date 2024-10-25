clear all
close all

load("cw1/data/cw1a.mat")

xs = linspace(-3, 3, 1001)';

meanfunc = []; hyp_init.mean = [];
covfunc = @covPeriodic; hyp_init.cov = [-1 0 0];
likfunc = @likGauss; hyp_init.lik = 0;
hyp_opt = minimize(hyp_init, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% Display outputs
Z_opt = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

fprintf('Optimised hyper-paramters:\n')
disp(structfun(@exp, hyp_opt, UniformOutput=false))
fprintf('Optimised log marginal likelihood: %f\n', -Z_opt)

% Plot predictive distribution
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];

figure;
hold on
fill([xs; flipdim(xs,1)], f, [7 7 7]/8, DisplayName='95% Prediction Error Bars')
plot(xs, mu, DisplayName='Prediction Mean'); 
scatter(x, y, '+', DisplayName='Data');
xlabel('Input - x')
ylabel('Output - y')
legend

saveas(gcf,'figures/C/periodic_covariance_plot','epsc')

% Calculate measurement error
mu_data = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x);

error = y - mu_data;
x_cdf = linspace(min(error), max(error), 1001);
y_cdf = normcdf(x_cdf, 0, exp(hyp_opt.lik(1)));

figure
hold on
cdfplot(error)
plot(x_cdf, y_cdf)
xlabel('Noise - \eta')
ylabel('F(\eta)')
legend('Empirical CDF', 'Theoretical CDF')

saveas(gcf,'figures/C/noise_cdf','epsc')

figure
scatter(x, error, '.')
xlabel('Input - X')
ylabel('Noise - \eta')

saveas(gcf,'figures/C/noise_plot','epsc')