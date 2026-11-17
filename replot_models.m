function replot_models()
% Export the three manuscript design-model panels using the original scripts.
root = fileparts(mfilename('fullpath'));
out = fullfile(root,'outputs','models'); if ~exist(out,'dir'), mkdir(out); end
original = pwd; clean = onCleanup(@() cd(original));
set(groot,'DefaultFigureVisible','off');
cd(fullfile(root,'figures','s01')); draw_resistance(); exportgraphics(gca,fullfile(out,'S1A.pdf'),'ContentType','vector');
cd(fullfile(root,'figures','s02')); draw_time(); exportgraphics(gca,fullfile(out,'S2A.pdf'),'ContentType','vector');
draw_lifetime(); exportgraphics(gca,fullfile(out,'S2B.pdf'),'ContentType','vector'); close all;
end
function draw_resistance()
GoldResistivity_fig1
end
function draw_time()
wirtingDesign
end
function draw_lifetime()
MTTF_tot_Au
end
