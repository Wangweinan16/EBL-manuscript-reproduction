function [data, out, meta] = compute_sites_per_shank_class_ranges_from_xlsx_v2()
% Compute per-class [xmin xmax; ymin ymax] ranges from the attached workbook.
%
% v2 change:
% - Plot inclusion is driven ONLY by the rows present in the workbook sheet
%   'PLOT_DATA_DEFAULT' (no additional filtering by DEVICE_DATA.include_default_plot).
%
% Goal:
% - X axis: cross-section per electrode (um^2/site)  -> column: x_value_um2_per_site
% - Y axis:
%     - Prefer using the plot sheet directly: PLOT_DATA_DEFAULT.y_value_sites_per_shank
%     - Otherwise fall back to computing channels_per_shank from DEVICE_DATA:
%           channels_per_shank = device_simultaneous_channels_total / resident_penetrants_per_device
%
% Outputs:
% - Prints a summary table (per class) with xmin/xmax/ymin/ymax and N devices.
% - Saves the computed `data` struct (and metadata) to a `.mat` file that the plotting script loads.
%
% Notes / assumptions:
% - resident_penetrants_per_device is treated as "shanks per device" for shank-array devices.
%   If it is missing or zero, we fall back to 1 to avoid divide-by-zero.
% - Class mapping is based on PLOT_DATA_DEFAULT.class_name.
%

XLSX_PATH = fullfile(fileparts(mfilename('fullpath')), ...
    'neural_probe_data_v3_sites_per_shank_values_only.xlsx');

MAT_PATH = fullfile(fileparts(mfilename('fullpath')), ...
    'channels_per_shank_class_ranges_v3_plot_data_default_only.mat');

plotd  = readtable(XLSX_PATH, 'Sheet', 'PLOT_DATA_DEFAULT');

use_plot_sheet_y = ismember('y_value_sites_per_shank', plotd.Properties.VariableNames);

if use_plot_sheet_y
    joined = plotd;
    joined.y_value_channels_per_shank = joined.y_value_sites_per_shank;
else
    device = readtable(XLSX_PATH, 'Sheet', 'DEVICE_DATA');
    % Compute channels per shank from DEVICE_DATA
    pen = device.resident_penetrants_per_device;
    pen(isnan(pen) | pen == 0) = 1;
    device.channels_per_shank = device.device_simultaneous_channels_total ./ pen;
    % Join plot sheet with channels_per_shank by (class_id, device_name)
    joined = outerjoin(plotd, device(:, {'class_id','device_name','channels_per_shank'}), ...
        'Keys', {'class_id','device_name'}, 'MergeKeys', true, 'Type', 'left');
    joined.y_value_channels_per_shank = joined.channels_per_shank;
end

% Define class mapping and class subset (screenshot-style core classes)
orig = { ...
    'Utah arrays'
    'Passive silicon shank arrays'
    'E-beam / nanofabricated passive silicon shank arrays'
    'Active CMOS silicon shank arrays'
    'Carbon-fiber ultramicroelectrode arrays'
    'Planar polymer shank arrays'
    'Ultraflexible lithographic polymer threads'
    'NET-EBL (this work)'
    };
short = { ...
    'utah arrays'
    'silicon shank'
    'e-beam silicon shank'
    'neuropixels'
    'carbon fiber'
    'planar polymer shank'
    'PL NET'
    'NET-EBL'
    };
map = containers.Map(orig, short);

class_names = joined.class_name;
if istable(class_names)
    class_names = table2cell(class_names);
end
if isstring(class_names)
    class_names = cellstr(class_names);
elseif ischar(class_names)
    class_names = cellstr(class_names);
end

% v2: keep only PLOT_DATA_DEFAULT rows for the core classes
is_keep = ismember(class_names, orig);
joined = joined(is_keep, :);

% Apply mapping
class_names = joined.class_name;
if istable(class_names)
    class_names = table2cell(class_names);
end
if isstring(class_names)
    class_names = cellstr(class_names);
elseif ischar(class_names)
    class_names = cellstr(class_names);
end

num_joined = size(joined, 1);
class_short = cell(num_joined, 1);
for i = 1:num_joined
    value_i = class_names{i};
    if isstring(value_i)
        value_i = char(value_i);
    end
    class_short{i} = map(value_i);
end
joined.class_short = class_short;

% Aggregate per class
classes = unique(joined.class_short, 'stable');
out = table('Size', [numel(classes) 6], ...
    'VariableTypes', {'cell','double','double','double','double','double'}, ...
    'VariableNames', {'class_short','x_min','x_max','y_min','y_max','n'});

for ci = 1:numel(classes)
    c = classes{ci};
    sub = joined(strcmp(joined.class_short, c), :);

    x = sub.x_value_um2_per_site;
    y = sub.y_value_channels_per_shank;

    out.class_short{ci} = c;
    out.x_min(ci) = min(x);
    out.x_max(ci) = max(x);
    out.y_min(ci) = min(y);
    out.y_max(ci) = max(y);
    out.n(ci) = size(sub, 1);
end

disp(out);

% Build a MATLAB struct matching the plotting script's expectation:
% data.<class_key> = [x_min x_max; y_min y_max]
data = struct();
num_classes = size(out, 1);
for ci = 1:num_classes
    key = strrep(out.class_short{ci}, ' ', '_');
    key = strrep(key, '-', '_');
    data.(key) = [out.x_min(ci), out.x_max(ci); out.y_min(ci), out.y_max(ci)];
end

% Save for the plotting script
meta = struct();
meta.xlsx_path = XLSX_PATH;
meta.mat_path = MAT_PATH;
meta.generated_on = datestr(now);
if use_plot_sheet_y
    meta.y_definition = 'PLOT_DATA_DEFAULT.y_value_sites_per_shank (treated as channels_per_shank)';
else
    meta.y_definition = 'channels_per_shank = device_simultaneous_channels_total / resident_penetrants_per_device';
end
meta.plot_rows_source = 'PLOT_DATA_DEFAULT (no include_default_plot filtering)';
meta.class_mapping_source = 'PLOT_DATA_DEFAULT.class_name mapped via orig->short in this script';
save(MAT_PATH, 'data', 'out', 'meta');
fprintf('\nSaved: %s\n', MAT_PATH);

% Print MATLAB struct lines
fprintf('\nMATLAB struct lines (paste into data = struct(...)):\n');
for ci = 1:num_classes
    key = strrep(out.class_short{ci}, ' ', '_');
    key = strrep(key, '-', '_');
    fprintf('    ''%s'', [%.10g, %.10g; %.10g, %.10g], ...\n', ...
        key, out.x_min(ci), out.x_max(ci), out.y_min(ci), out.y_max(ci));
end

end

