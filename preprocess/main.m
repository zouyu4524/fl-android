%% main

%% load from raw txt file, and save results to an intermediate .mat file
% Because the loading process takes quite a long time (about 10 mins), 
% this part is commentted out.

% filename = 'raw/WISDM_ar_v1.1_raw.txt';
% [xyz_matrix, ID_matrix, labels] = txt2matrix(filename);
% save('intermediate/matrics.mat');

%% load from .mat file, then stack raw data into entries and save as csv file
% the dataset is separated into train set and test set, the partition ratio
% can be adjusted, as well as the window size and shuffle property in
% script 'matrix2csv'.
% After running matrix2csv, two .csv files will be generated and store
% under path 'csv'.
matrix2csv;