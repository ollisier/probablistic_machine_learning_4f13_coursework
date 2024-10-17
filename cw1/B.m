clear all
close all

load("cw1/data/cw1a.mat")

meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;


% Sweep hyper-parameters and plot slices 
[ell, sf, sn] = meshgrid(-3:0.1:3, -1:0.01:1, -3:0.1:0);

nlZ = zeros(size(ell));

for i = 1:numel(ell)
    hyp = struct();
    hyp.mean = [];
    hyp.cov = [ell(i) sf(i)];
    hyp.lik = sn(i);

    nlZ(i) = log(gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y));
end

 
figure

x_figs = ceil(sqrt(size(ell, 3)));
y_figs = ceil(size(ell, 3) / x_figs);

for i = 1:size(ell, 3)
    subplot(y_figs, x_figs, i);
    contour(ell(:,:,i), sf(:,:,i), nlZ(:,:,i), 30)
    title(sprintf('sn = %f', sn(1,1,i)))
    xlabel('ell')
    ylabel('sf')
    clim([min(nlZ, [], 'all'), max(nlZ, [], 'all')])
end

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
    title(sprintf('ell = %f, sf = %f, sn = %f -> nlZ = %f', hyp_min.cov(1), hyp_min.cov(2), hyp_min.lik(1), nlZ_min))
end