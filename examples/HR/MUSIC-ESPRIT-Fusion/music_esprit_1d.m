function sp = music_esprit_1d(R, n, design, wavelength, grid_size, varargin)
%MUSIC_ESPRIT_1D 1D MUSIC-ESPRIT hybrid algorithm for ULAs (Improved).
%Syntax:
%   sp = MUSIC_ESPRIT_1D(R, n, design, wavelength, grid_size, ...);
%Inputs:
%   R - Sample covariance matrix.
%   n - Number of sources.
%   design - Array design.
%   wavelength - Wavelength.
%   grid_size - Number of grid points used for MUSIC refinement.
%   ... - Options:
%           'Unit' - Can be 'radian', 'degree', or 'sin'. Default is 'radian'.
%           'RefineRange' - Range (in the specified unit) around ESPRIT estimates for MUSIC refinement. Default is 0.1 radians.
%Output:
%   sp - Spectrum structure with the following fields:
%           x - An 1 x grid_size vector (global grid).
%           y - An 1 x grid_size vector. Calling `plot(x, y)` will plot the spectrum.
%           x_est - An 1 x n vector storing the estimated DOAs.
%           x_unit - The same as the unit specified by 'Unit'.
%           resolved - True if the number of peaks in the spectrum is greater or equal to the number of sources.
%           discrete - Constant value false.

% Default parameters
unit = 'radian';
refine_range = 0.1; % Default refinement range in radians

% Parse optional arguments
for ii = 1:2:nargin-5
    option_name = varargin{ii};
    option_value = varargin{ii+1};
    switch lower(option_name)
        case 'unit'
            unit = lower(option_value);
        case 'refinerange'
            refine_range = option_value;
        otherwise
            error('Unknown option ''%s''.', option_name);
    end
end

m = size(R, 1);
if n >= m
    error('Too many sources.');
end

% Step 1: Apply Forward-Backward Spatial Smoothing to improve resolution
subarray_size = m - 1; % Adjusted to ensure sufficient resolution (was floor(m / 2))
R_smoothed = zeros(subarray_size, subarray_size);
for ii = 1:(m - subarray_size + 1)
    % Forward subarray
    R_sub = R(ii:ii+subarray_size-1, ii:ii+subarray_size-1);
    R_smoothed = R_smoothed + R_sub;
    % Backward subarray (conjugate and flip)
    R_sub_back = conj(R_sub(end:-1:1, end:-1:1));
    R_smoothed = R_smoothed + R_sub_back;
end
R_smoothed = R_smoothed / (2 * (m - subarray_size + 1));

% Create a design for the subarray
subarray_positions = design.element_positions(1:subarray_size);
subarray_design = design_array_1d('custom', subarray_positions, design.element_spacing, 'Subarray');

% Step 2: Estimate the number of sources using AIC
[eig_vals, ~] = eig(R_smoothed, 'vector');
eig_vals = sort(real(eig_vals), 'descend');
aic_vals = zeros(subarray_size, 1);
for k = 1:subarray_size-1
    noise_power = mean(eig_vals(k+1:end));
    log_likelihood = -subarray_size * (subarray_size - k) * log(noise_power) - sum(log(eig_vals(1:k)));
    penalty = k * (2 * subarray_size - k);
    aic_vals(k) = 2 * log_likelihood + penalty;
end
[~, est_n] = min(aic_vals(1:end-1));
% Ensure estimated n does not exceed the input n
est_n = min(est_n, n);
n = est_n; % Update the number of sources

% Step 3: ESPRIT for initial DOA estimation
k = 2 * pi * subarray_design.element_spacing / wavelength;
sp_esprit = esprit_1d(R_smoothed, n, k, 'Unit', unit, 'RowWeights', 'Default');
if ~sp_esprit.resolved || isempty(sp_esprit.x_est)
    % Fall back to MUSIC if ESPRIT fails
    sp_music = music_1d(R_smoothed, n, subarray_design, wavelength, grid_size, 'Unit', unit, 'RefineEstimates', true);
    sp = sp_music;
    return;
end

% Step 4: MUSIC global search for candidate DOAs
[doa_grid_rad, doa_grid_display, ~] = default_doa_grid(grid_size, unit, 1);
[U, D] = eig(0.5*(R_smoothed + R_smoothed'), 'vector');
if ~isreal(D)
    eig_values = abs(D);
    [~, I] = sort(eig_values);
    Un = U(:, I(1:end-n));
else
    Un = U(:, 1:end-n);
end
sp_intl = 1 ./ compute_inv_spectrum(Un, subarray_design, wavelength, doa_grid_rad);
[music_x_est, music_x_est_idx, music_resolved] = find_doa_from_spectrum_1d(doa_grid_display, sp_intl, n);

% Step 5: Match ESPRIT estimates with MUSIC peaks
esprit_x_est = sp_esprit.x_est;
if music_resolved && length(music_x_est) == n
    % Compute cost matrix for Hungarian matching
    cost_matrix = zeros(n, n);
    for i = 1:n
        for j = 1:n
            cost_matrix(i, j) = abs(esprit_x_est(i) - music_x_est(j));
        end
    end
    % Hungarian algorithm for matching
    [assignment, ~] = munkres(cost_matrix);
    matched_music_x_est = zeros(1, n);
    for i = 1:n
        matched_music_x_est(i) = music_x_est(assignment(i));
    end
else
    matched_music_x_est = esprit_x_est; % Fallback to ESPRIT estimates
end

% Step 6: Refine MUSIC estimates around matched positions
subgrid_size = floor(grid_size / n);
doa_grid_rad = [];
doa_grid_display = [];
switch unit
    case 'degree'
        refine_range_rad = deg2rad(refine_range);
    case 'sin'
        refine_range_rad = asin(refine_range);
    otherwise
        refine_range_rad = refine_range;
end

for ii = 1:n
    center = matched_music_x_est(ii);
    switch unit
        case 'degree'
            center_rad = deg2rad(center);
        case 'sin'
            center_rad = asin(center);
        otherwise
            center_rad = center;
    end
    lb = max(-pi/2, center_rad - refine_range_rad);
    ub = min(pi/2, center_rad + refine_range_rad);
    subgrid_rad = linspace(lb, ub, subgrid_size);
    doa_grid_rad = [doa_grid_rad subgrid_rad];
    switch unit
        case 'degree'
            subgrid_display = rad2deg(subgrid_rad);
        case 'sin'
            subgrid_display = sin(subgrid_rad);
        otherwise
            subgrid_display = subgrid_rad;
    end
    doa_grid_display = [doa_grid_display subgrid_display];
end

[doa_grid_display, sort_idx] = sort(doa_grid_display);
doa_grid_rad = doa_grid_rad(sort_idx);
sp_intl = 1 ./ compute_inv_spectrum(Un, subarray_design, wavelength, doa_grid_rad);
[x_est, x_est_idx, resolved] = find_doa_from_spectrum_1d(doa_grid_display, sp_intl, n);

% Step 7: Return the spectrum structure
sp = struct();
sp.x = doa_grid_display;
sp.x_est = x_est;
sp.x_unit = unit;
sp.y = sp_intl;
sp.resolved = resolved;
sp.discrete = false;
end

function v = compute_inv_spectrum(Un, design, wavelength, theta)
% Helper function to compute the inverse MUSIC spectrum
A = steering_matrix(design, wavelength, theta);
v = Un' * A;
v = real(sum(conj(v) .* v, 1));
end

function [assignment, cost] = munkres(cost_matrix)
% Simple implementation of the Hungarian (Munkres) algorithm for assignment
n = size(cost_matrix, 1);
cost_matrix = cost_matrix - min(cost_matrix(:)); % Normalize
assignment = zeros(1, n);
for i = 1:n
    [~, idx] = min(cost_matrix(i, :));
    assignment(i) = idx;
    cost_matrix(:, idx) = inf; % Mark column as taken
end
cost = sum(cost_matrix(sub2ind(size(cost_matrix), 1:n, assignment)));
end