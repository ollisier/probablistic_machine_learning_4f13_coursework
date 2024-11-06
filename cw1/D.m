clear all
close all

seed = 2;
N_points = 200; N_samples = 2;
x = linspace(-5,5,N_points)';

cov_funcs = {
    {@covPeriodic},
    {@covSEiso},
    {@covProd, {@covPeriodic, @covSEiso}},
};

cov_hyps = {
    [-0.5 0 0],
    [2 0],
    [-0.5 0 0 2 0]
};

for i = 1:length(cov_funcs)
    K = feval(cov_funcs{i}{:}, cov_hyps{i}, x);
    y = chol(K + 1e-6*eye(N_points))' * gpml_randn(seed, N_points, N_samples);
    figure
    plot(x, y)
    xlabel('Input - X')
    ylabel('Output - Y')
    saveas(gcf,sprintf('figures/D/prod_kernel_samples_%d', i),'epsc')
end
