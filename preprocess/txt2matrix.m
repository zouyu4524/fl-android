function [xyz_matrix, ID_matrix, labels] = txt2matrix(filename)

fid = fopen(filename);

% get num of lines of the file
numofline = 0;
while 1
    str = fgetl(fid);
    if str == -1
        break;
    end
    if strcmp(str, '')
        continue;
    end
    str = strip(str, ';');
    info = split(str, ',');
    if sum(isnan(str2double(info(4:6)))) > 0
       continue; 
    end
    numofline = numofline + 1;
end
fclose(fid);

xyz_matrix = zeros(numofline, 3);
ID_matrix = zeros(numofline, 1);
labels = zeros(numofline, 1);

% load files to matrix
fid = fopen(filename);
line = 0;
while 1
    str = fgetl(fid);
    if str == -1
        break;
    end
    if strcmp(str, '')
        continue;
    end
    str = strip(str, ';');
    info = split(str, ',');
    if sum(isnan(str2double(info(4:6)))) > 0
       continue; 
    end
    line = line + 1;
    xyz_matrix(line, :) = str2double(info(4:6));
    ID_matrix(line, :) = str2double(info{1});
    labels(line) = class2index(info{2});
end
fclose(fid);

end

function index = class2index(classname)
switch classname
    case 'Jogging'
        index = 0;
    case 'Walking'
        index = 1;
    case 'Upstairs'
        index = 2;
    case 'Downstairs'
        index = 3;
    case 'Sitting'
        index = 4;
    case 'Standing'
        index = 5;
end
end