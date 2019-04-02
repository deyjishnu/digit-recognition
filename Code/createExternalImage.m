function createExternalImage(row_num, imagename)
% CREATEEXTERNALIMAGE Creates, displays and saves an image with the name 'imagename.extension' using the
% pixels of the 'row_num'-th image of the MNIST Training Dataset

%% Initialization

if ~exist('row_num', 'var') || isempty(row_num)
    row_num = 1;
end

if ~exist('imagename', 'var') || isempty(imagename)
    imagename = ['Image_MNIST_Training_Row_' num2str(row_num) '.jpg'];
end

% Load the data from MNIST Dataset
data = load('MNISTDataset.mat');
a = data.trainingImages(row_num, :);

fprintf('Displaying and saving requested image ...\n');

% Reshape the row vector to a 28*28 pixel image
a = reshape(a, 28, 28);
imshow(a);

% Save the image
imwrite(mat2gray(a),imagename);

% Find information about the image
% info = imfinfo(imagename);
% disp(info.ColorType);

fprintf('Image saved with the name: %s', imagename);
end