function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'r', 'b');
if(fp == -1)
    error('Could not open file');
end
    
header = fread(fp, 1, 'int32');
if header ~= 2049
    error('Invalid label file header');
end

numLabels = fread(fp, 1, 'int32');

labels = fread(fp, inf, 'unsigned char');

if(size(labels,1) ~= numLabels)
    error('Mismatch in label count');
end

fclose(fp);

end