clc;
clear all;

%%%%%%%%%%%%%%%%%%% displays  a picture from the mnist data set
data=load('MNIST_All(Dataset).mat');
b=data.train4(1520,:);
b=reshape(b,28,28);
imshow(b);figure;

imwrite(mat2gray(b),'Name.jpg');
info = imfinfo('Name.jpg');
disp(info.ColorType);
%c=imread('Name.jpg');
% figure;


%%%%%%%%%%%%%%%%%%% displays an external scanned image
basename='nine';
imagefile=[basename '.jpg'];    
info = imfinfo(imagefile);
disp(info.ColorType);
a=imread(imagefile);
a=imresize(a,[28 28]);
a=rgb2gray(a);
imshow(a);figure  %%//normal display
c=reshape(a,1,size(a,1)*size(a,2));
for i=1:784
    c(1,i)=255-c(1,i);
end
c=reshape(c,28,28);
imshow(c); %%//after subtracting  255.. 

%%but what to do if image is black and white????
%%we can use if else
%%relace this code with the second one above to check if working properly.
