clear all
close all

load("cw1/data/cw1e.mat")

figure
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11))
xlabel('X_1')
ylabel('X_2')
zlabel('Y')

saveas(gcf,'figures/E/raw_data','epsc')

[xs1, xs2] = meshgrid(linspace(-10, 10, 101), linspace(-10, 10, 101));
xs = [reshape(xs1, [], 1), reshape(xs2, [], 1)];

mean_func = []; hyp_init.mean = [];
lik_func = @likGauss; hyp_init.lik = 0;

cov_funcs = {
    {@covSEard},
    {@covSum, {@covSEard, @covSEard}},
};

cov_hyp = {
    0.1*randn(3,1),
    0.1*randn(6,1),
};

cov_name = {
    '@covSEard', 
    '@covSum, \{@covSEard, @covSEard\}'
};

for i = 1:length(cov_funcs)
    hyp = hyp_init;
    hyp.cov = cov_hyp{i};
    hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_funcs{i}, lik_func, x, y);
    Z_opt = gp(hyp_opt, @infGaussLik, mean_func, cov_funcs{i}, lik_func, x, y);
    fprintf(cov_name{i})
    fprintf('Hyper-paramters:\n')
    fields = fieldnames(hyp_opt);
    for j = 1:length(fields)
        field = fields{j};
        fprintf('%s: %s\n', field, mat2str(round(exp(hyp_opt.(field)), 4)))
    end
    fprintf('Log marginal likelihood: %f\n', -Z_opt)

    [mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_funcs{i}, lik_func, x, y, xs);

    figure
    hold on
    surf(xs1, xs2, reshape(mu + 2*s2, size(xs1)))
    surf(xs1, xs2, reshape(mu - 2*s2, size(xs1)))

    view([-37.5 30])
    xlabel('X_1')
    ylabel('X_2')
    zlabel('Y')

    saveas(gcf,sprintf('figures/E/kernel_comparison_%d', i),'epsc')

end