clear all
close all

load("cw1/data/cw1a.mat")

xs = linspace(-3, 3, 1001)';

% Main Code
meanfunc = []; hyp_init.mean = [];
covfunc = @covSEiso; hyp_init.cov = [-1 0];
likfunc = @likGauss; hyp_init.lik = 0;
hyp_opt = minimize(hyp_init, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% Display outputs
Z_init = gp(hyp_init, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
Z_opt = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

fprintf('Initial hyper-paramters:\n')
disp(structfun(@exp, hyp_init, UniformOutput=false))
fprintf('Initial negative log marginal likelihood: %f\n', Z_init)

fprintf('Optimised hyper-paramters:\n')
disp(structfun(@exp, hyp_opt, UniformOutput=false))
fprintf('Optimised negative log marginal likelihood: %f\n', Z_opt)

% Plot predictive distribution
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];

fig = figure;
hold on
fill([xs; flipdim(xs,1)], f, [7 7 7]/8, DisplayName='95% Prediction Error Bars')
plot(xs, mu, DisplayName='Prediction Mean'); 
scatter(x, y, '+', DisplayName='Data');
xlabel('Input - x')
ylabel('Output - y')
legend

fig.Position = [0,0,800,420];

saveas(fig,'figures/A/initial_fit_plot','epsc')