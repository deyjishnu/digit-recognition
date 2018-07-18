function p = predictExternalImage(imagefile)
% PREDICTEXTERNALIMAGE Predicts what digit is contained in an external
% image given a trained neural network
% p = PREDICTEXTERNALIMAGE(imagefile) outputs the predicted label of an image 

%% Initialization

if ~exist('imagefile', 'var') || isempty(imagefile)
    error('Pass an image file as an argument');
end

fprintf('Displaying 28-by-28 grayscale image: %s ...\n', imagefile);

% Find information about the image
% info = imfinfo(imagefile);
% disp(info.ColorType);

% Read RGB image
a = imread(imagefile);

% Resize image to a 28*28 pixel image
a = imresize(a,[28 28]);

% Convert the RGB image to grayscale
a = rgb2gray(a);

% Display black image on white background
% imshow(a);figure

% Display white image on black background
a = 255 - a;
imshow(a);

% Feature Scaling to match MNIST Data
a = double(a)/255;

% Convert to a row vector
a = a(:)';

% Load .mat file containing Theta1 and Theta2
load('neuralNetworkWeights.mat');

% Predict the label of the image using the 'predict' function
p = predict(Theta1, Theta2, a);

% Display the predicted digit on the console
fprintf('Predicted digit: %.0f\n', mod(p,10));
end