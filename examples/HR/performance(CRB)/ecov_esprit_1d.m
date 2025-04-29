function C = ecov_esprit_1d(design, wavelength, doas, p, noise_var, snapshot_count, mode)
%ECOV_ESPRIT_1D Asymptotic covariance matrix of ESPRIT DOA estimation errors for ULAs.
%Syntax:
%   C = ECOV_ESPRIT_1D(design, wavelength, doas, p, noise_var, snapshot_count);
%   C = ECOV_ESPRIT_1D(design, wavelength, doas, p, noise_var, snapshot_count, 'DiagonalsOnly');
%Inputs:
%   design - Array design (must be ULA).
%   wavelength - Wavelength.
%   doas - DOA vector in radians.
%   p - Source power vector or scalar.
%   noise_var - Noise power.
%   snapshot_count - Number of snapshots.
%   mode - 'Full' for full covariance matrix, 'DiagonalsOnly' for variances.
%Output:
%   C - Covariance matrix or variance vector.
if design.dim ~= 1 || ~strcmpi(design.type, 'ula')
    error('ULA expected.');
end
if nargin <= 6
    mode = 'full';
end
m = design.element_count;
k = length(doas);
p = unify_source_power_vector(p, k);
ds = 1; % default displacement, as in esprit_1d
k_val = 2*pi*design.element_spacing/wavelength; % k for DOA derivative

% check source number
if k >= m - ds
    error('Too many sources.');
end

% compute ideal covariance matrix
A = steering_matrix(design, wavelength, doas);
R_ideal = A * diag(p) * A' + noise_var * eye(m);

% eigen decomposition
[E, D] = eig(0.5*(R_ideal + R_ideal'), 'vector');
[~, idx] = sort(D, 'descend');
E = E(:, idx);
D = D(idx);
Us = E(:, 1:k); % signal subspace
Un = E(:, k+1:end); % noise subspace
Es = Us; % signal subspace for ESPRIT
Es1 = Es(1:end-ds, :);
Es2 = Es(ds+1:end, :);

% compute Ws matrix
Ws = zeros(k, k);
for i = 1:k
    denom = D(i) - noise_var;
    if denom < 1e-6 % numerical stability
        denom = 1e-6;
    end
    Ws(i,i) = (D(i) * noise_var) / (denom^2);
end
T = Us * Ws * Us';

% compute rho_k (simplified, consistent with esprit_1d LS formulation)
rho = pinv(Es1);

% compute h(beta_k) = (rho_k^H rho_k) d(beta_k)^H Un Un^H d(beta_k)
h = zeros(k, 1);
for i = 1:k
    rho_i = rho(:, i);
    rho_term = rho_i' * rho_i;
    % compute derivative d(beta_k)
    indices = 0:(m-1);
    d_i = 1j * k_val * cos(doas(i)) * indices' .* A(:, i);
    proj = Un' * d_i;
    h(i) = rho_term * (proj' * proj);
end

% compute variance (diagonal elements only, as per Theorem 6.5.2)
C = zeros(k, k);
for i = 1:k
    a_i = A(:, i);
    term = a_i' * T * a_i;
    C(i,i) = (1 / (2 * snapshot_count)) * real(term) / h(i);
end

% output
switch lower(mode)
    case 'diagonalsonly'
        C = diag(C);
    case 'full'
        C = C; % only diagonal elements are computed as per theorem
    otherwise
        error('Invalid mode.');
end