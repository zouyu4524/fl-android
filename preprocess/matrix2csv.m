%% matrix to csv

windowsize = 90;
windowsize = ceil(windowsize/2)*2;

train_ratio = 0.7;
shuffle = true;

segmentation;
MATRICS = cell(size(separations, 1), 1);

for k = 1 : size(separations, 1)
    index = separations(k, :);
    index = index(1) : index(2);
    moi = xyz_matrix(index, :);
    loi = labels(index);
    numofentry = floor(size(moi, 1)/(windowsize/2)) - 1;
    MATRICS{k} = zeros(numofentry, windowsize*3+1);
    for j = 1 : numofentry
        ioi = (j-1)*windowsize/2+1 : (j+1)*windowsize/2;
        MATRICS{k}(j, 1) = mode(loi(ioi));
        MATRICS{k}(j, 2:end) = reshape(moi(ioi, :)', 1, windowsize*3);
    end
end

%% concatenate MATRICS into MATRIX
MATRIX = [];
for k = 1 : length(MATRICS)
    if isempty(MATRICS{k})
        continue;
    end
    MATRIX = [MATRIX; MATRICS{k}];
end

%% separate into train and test set
numoftrain = round(train_ratio*size(MATRIX, 1));
if shuffle
    seq = randperm(size(MATRIX, 1));
else
    seq = 1:size(MATRIX, 1);
end
seqoftrain = seq(1:numoftrain);
seqoftest = seq(numoftrain+1:end);

% write to csv
csvwrite('csv/WISDM_ar_train.csv', MATRIX(seqoftrain,:));
csvwrite('csv/WISDM_ar_test.csv', MATRIX(seqoftest,:));
