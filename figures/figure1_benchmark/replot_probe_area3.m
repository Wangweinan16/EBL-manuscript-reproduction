function [fig_handle, plot_handles, data] = replot_probe_area3()
% MATLAB-only copy for regenerating probe_area3.svg.
%
% This file intentionally preserves the older MATLAB/Batik ellipse-plot
% style used by `code_for_area_channels_per_shank_vs_cross_section_v2.m`.
% It is a copy with only the requested publication-view changes:
%
% 1) x-axis limits are expanded to [3e-2, 2e4].
% 2) y-axis limits are expanded to [2e-1, 3e3].
% 3) Utah arrays and carbon fiber get a small visual-only y range so their
%    otherwise collapsed 1-to-1 channel/shank ranges can be seen.
% 4) Utah arrays also get a small visual-only x range so their otherwise
%    collapsed x range can be seen.
% 5) All class envelope contours are dashed except NET-EBL, which remains
%    solid.
%
% Data provenance:
% - Source workbook:
%     neural_probe_data_v3_sites_per_shank_values_only.xlsx
% - Range file loaded by this script:
%     channels_per_shank_class_ranges_v3_plot_data_default_only.mat
% - The MAT file contains one 2-by-2 range matrix per class:
%     row 1: [x_min x_max], cross-section per electrode (um^2)
%     row 2: [y_min y_max], channels per shank
%
% To regenerate the source range file from the workbook, run:
%     compute_sites_per_shank_class_ranges_from_xlsx_v2
%
% To regenerate probe_area3.svg from the range file, run this function:
%     replot_probe_area3

% Output and requested axis controls.
OUTPUT_SVG = fullfile(fileparts(mfilename('fullpath')), 'probe_area3.svg');
X_LIMITS = [3e-2 2e4];
Y_LIMITS = [2e-1 3e3];

% Visual-only y range for classes whose real y range is [1 1].
% Because the axis is log, this is applied as log10(y).
LOW_Y_RANGE_FIELDS = {'utah_arrays', 'carbon_fiber'};
LOW_Y_LOG10_RANGE = [-0.12 0.12];

% Visual-only x range for Utah arrays, whose real x range is also collapsed.
% This applies a symmetric log10 span around the original Utah x value.
LOW_X_RANGE_FIELDS = {'utah_arrays'};
LOW_X_LOG10_RANGE = [-0.12 0.12];

% Custom dashed contour settings for non-NET-EBL class envelopes.
% Smaller numbers make shorter dashes and smaller gaps.
DASH_POINTS = 40;
GAP_POINTS = 30;

% Load precomputed ranges.
MAT_PATH = fullfile(fileparts(mfilename('fullpath')), ...
    'channels_per_shank_class_ranges_v3_plot_data_default_only.mat');
if ~isfile(MAT_PATH)
    error(['Missing required MAT file: ' MAT_PATH newline ...
           'Run compute_sites_per_shank_class_ranges_from_xlsx_v2 first.']);
end
s = load(MAT_PATH, 'data', 'meta');
data = s.data;

% Marker styles corresponding to each class.
markers = {'s', 'x', '*', 'o', '+', 'v', 'd', 'p', 'h'};
fallback_colors = lines(length(markers));

fig_handle = figure;
hold on;
fields = fieldnames(data);
plotHandles = gobjects(length(fields), 1);

for k = 1:length(fields)
    field_name = fields{k};
    range = data.(field_name);
    color = color_for_field(field_name, fallback_colors(k, :));

    % Expand only the displayed y envelope for Utah arrays and carbon fiber.
    % This makes their range visible without editing the MAT file.
    if ismember(field_name, LOW_Y_RANGE_FIELDS)
        range(2, :) = 10 .^ LOW_Y_LOG10_RANGE;
    end
    if ismember(field_name, LOW_X_RANGE_FIELDS)
        x_center_raw = sqrt(prod(range(1, :)));
        range(1, :) = x_center_raw .* (10 .^ LOW_X_LOG10_RANGE);
    end

    x_center = mean(range(1, :));
    y_center = mean(range(2, :));
    ellipse_width = diff(range(1, :));
    ellipse_height = diff(range(2, :));

    if strcmp(field_name, 'NET_EBL')
        contour_line_style = '-';
    else
        contour_line_style = '--';
    end

    draw_ellipse( ...
        x_center, ...
        y_center, ...
        ellipse_width, ...
        ellipse_height, ...
        color, ...
        0.5, ...
        contour_line_style, ...
        DASH_POINTS, ...
        GAP_POINTS);

    plotHandles(k) = scatter(x_center, y_center, 25, ...
        'Marker', markers{k}, ...
        'MarkerEdgeColor', color, ...
        'LineWidth', 2);
end

set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'LineWidth', 0.25);
xlabel('Cross-section per electrode (um^2)', 'FontSize', 12);
ylabel('Channels per shank', 'FontSize', 12);

labels = cell(size(fields));
for k = 1:numel(fields)
    labels{k} = legend_label_for_field(fields{k});
end
lgd = legend(plotHandles, labels, 'Location', 'southeast', 'Interpreter', 'none');
set(lgd, 'Box', 'on');
if isprop(lgd, 'ItemTokenSize')
    lgd.ItemTokenSize = [8 8];
end

xlim(X_LIMITS);
ylim(Y_LIMITS);
grid off;
box on;
hold off;
plot_handles = plotHandles;

% MATLAB R2022a exports through Batik here, matching the existing SVG family.
saveas(fig_handle, OUTPUT_SVG);

end

function draw_ellipse(x_center, y_center, width, height, color, line_width, line_style, dash_points, gap_points)
    theta = linspace(0, 2*pi, 500);
    x = (width/2) * cos(theta) + x_center;
    y = (height/2) * sin(theta) + y_center;

    if strcmp(line_style, '--')
        fill(x, y, color, ...
            'FaceAlpha', 0.2, ...
            'EdgeColor', 'none');
        draw_short_dashed_outline(x, y, color, line_width, dash_points, gap_points);
    else
        fill(x, y, color, ...
            'FaceAlpha', 0.2, ...
            'EdgeColor', color, ...
            'LineWidth', line_width, ...
            'LineStyle', line_style);
    end
end

function draw_short_dashed_outline(x, y, color, line_width, dash_points, gap_points)
    step_points = dash_points + gap_points;
    n = numel(x);
    for start_idx = 1:step_points:n
        end_idx = min(start_idx + dash_points - 1, n);
        plot(x(start_idx:end_idx), y(start_idx:end_idx), ...
            'Color', color, ...
            'LineWidth', line_width);
    end
end

function color = color_for_field(field, fallback_color)
    % Publication palette grouped by probe family:
    % - PL NET and NET-EBL share the most salient NET-family crimson.
    % - Planar polymer uses green to broaden the palette.
    % - Silicon shank, e-beam silicon, and Neuropixels use related blues.
    % - Carbon fiber remains a flexible/non-silicon purple.
    % - Utah arrays are distinct but closer to the silicon family.
    switch field
        case {'NET_EBL', 'PL_NET', 'PL_NET_probe'}
            color = [0.75 0.07 0.12];      % #C1121F, salient NET crimson
        case 'planar_polymer_shank'
            color = [0.35 0.63 0.31];      % #59A14F, polymer green
        case 'carbon_fiber'
            color = [0.48 0.17 0.75];      % #7B2CBF, flexible purple
        case 'silicon_shank'
            color = [0.12 0.47 0.71];      % #1F77B4, silicon blue
        case 'e_beam_silicon_shank'
            color = [0.31 0.64 0.82];      % #4FA3D1, light silicon blue
        case 'neuropixels'
            color = [0.04 0.24 0.36];      % #0B3C5D, dark silicon blue
        case 'utah_arrays'
            color = [0.09 0.64 0.72];      % #17A2B8, rigid teal-blue
        otherwise
            color = fallback_color;
    end
end

function label = legend_label_for_field(field)
    % Map internal struct keys to publication labels.
    m = containers.Map( ...
        {'utah_arrays', 'silicon_shank', 'e_beam_silicon_shank', ...
         'neuropixels', 'carbon_fiber', 'planar_polymer_shank', ...
         'PL_NET', 'PL_NET_probe', 'NET_EBL'}, ...
        {'Utah arrays', 'Silicon shank', 'E-beam silicon shank', ...
         'Neuropixels', 'Carbon fiber', 'Planar polymer shank', ...
         'PL NET', 'PL NET', 'NET-EBL'});

    if isKey(m, field)
        label = m(field);
        return;
    end

    label = strrep(field, '_', ' ');
    if ~isempty(label)
        label(1) = upper(label(1));
    end
end
