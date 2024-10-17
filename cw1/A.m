clear all
close all

load("cw1/data/cw1a.mat")

meanfunc = []; hyp.mean = [];
covfunc = @covSEiso; hyp.cov = [-1 0];
likfunc = @likGauss; hyp.lik = 0;

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

disp(hyp2)


xs = linspace(-3, 3, 1001)';

[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];

figure(1)
hold on
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
plot(xs, mu); 
scatter(x, y, '+');