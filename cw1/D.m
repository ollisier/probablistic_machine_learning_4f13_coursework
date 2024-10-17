clear all
close all

N_points = 200;
N_samples = 10;

x = linspace(-5,5,N_points)';

meanfunc = []; hyp.mean = [];
cov_func = {@covProd, {@covPeriodic, @covSEiso}}; hyp.cov = [-0.5 0 0 2 0];

cov = feval(cov_func{:}, hyp.cov, x, x);
mean = meanZero(hyp.mean, x);

y = mean + chol(cov + 1e-6*eye(200)) * randn(N_points, N_samples);
figure
plot(x, y)
