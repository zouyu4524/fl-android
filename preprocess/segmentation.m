%% segmentation

load('intermediate/matrics.mat');
index = 1;

separations = [];

for k = 2 : length(ID_matrix)
    if ID_matrix(k) ~= ID_matrix(k-1)
       separations = [separations; [index, k-1]];
       index = k;
    end
end
separations = [separations; [index, length(ID_matrix)]];