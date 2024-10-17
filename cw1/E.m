clear all
close all

load("cw1/data/cw1e.mat")

figure

meanfunc = []; hyp_init.mean = [];
covfunc = {@covSum, {@covSEard, @covSEard}}; hyp_init.cov = 0.1*randn(6,1);
likfunc = @likGauss; hyp_init.lik = 0;

hyp_opt = minimize(hyp_init, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
disp(hyp_opt)

[Xs1, Xs2] = meshgrid(linspace(-3, 3, 101), linspace(-3, 3, 101));

[mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, [reshape(Xs1, [], 1), reshape(Xs2, [], 1)]);

figure
hold on
surf(Xs1, Xs2, reshape(mu, size(Xs1)))
scatter3(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11), '+')
