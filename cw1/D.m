clear all
close all

seed = 2;
N_points = 200; N_samples = 2;
x = linspace(-5,5,N_points)';
% cov_func = {@covProd, {@covPeriodic, @covSEiso}}; hyp.cov = [-0.5 0 0 2 0];
% K = feval(cov_func{:}, hyp.cov, x);
% y = chol(K + 1e-6*eye(N_points))' * gpml_randn(2, N_points, N_samples);

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

cov_names = {
    'Periodic SE (Small \lambda)',
    'SE (Large \lambda)',
    'Periodic SE (Small \lambda) * SE (Large \lambda)',
};

figure
for i = 1:length(cov_funcs)
    subplot(1, length(cov_funcs), i)
    K = feval(cov_funcs{i}{:}, cov_hyps{i}, x);
    y = chol(K + 1e-6*eye(N_points))' * gpml_randn(seed, N_points, N_samples);
    plot(x, y)
    xlabel('Input - X')
    title(cov_names{i})
end

f = gcf;
f.Position = [0, 0, 1500, 420];

subplot(1, length(cov_funcs), i)
ylabel('Output - Y')

saveas(gcf,'figures/D/prod_kernel_samples','epsc')
