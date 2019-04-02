function images = loadMNISTImages(filename)
% loadMNISTImages returns a [number of MNIST images]x28x28 matrix containing
% the raw MNIST images

fp = fopen(filename, 'r', 'b');
if(fp == -1)
    error('Could not open file');
end
    
header = fread(fp, 1, 'int32');
if header ~= 2051
    error('Invalid image file header');
end

numImages = fread(fp, 1, 'int32');
numRows = fread(fp, 1, 'int32');
numCols = fread(fp, 1, 'int32');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

images = images';

% Convert to double and rescale to [0,1]
images = double(images)/ 255;

end