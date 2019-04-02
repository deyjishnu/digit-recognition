%%the code for segmentation done in class by us..

clc;clear all;

%%fetching file 
[fname path]=uigetfile('*.*','enter an image');
fname=strcat(path,fname);

%reading into variable
c=imread(fname);

%2d conversion
c=rgb2gray(c);

%inverting image
c=~c; 

%measuring area of digits
se = strel('square',7);
im_close = imclose(c, se);
s = regionprops(im_close, 'BoundingBox');

%creating boxes around digits
bb = round(reshape([s.BoundingBox], 4, []).');
figure;
imshow(c);

%extracting boxes with individual digits
for idx = 1 : numel(s)
   rectangle('Position', bb(idx,:), 'edgecolor', 'red');
end

%string the segmented digits into array 
num = cell(1, numel(s));
for idx = 1 : numel(s)
    num{idx} = c(bb(idx,2):bb(idx,2)+bb(idx,4)-1, bb(idx,1):bb(idx,1)+bb(idx,3)-1);
end
figure;
imshow(num{5});

%writing a digit into a image file
imwrite(mat2gray(num{5}),'test1.jpg');
I=imread('test1.jpg');

%binary conversion
BW=imbinarize(I);
figure;
imshow(BW);

%filling up spaces
filled=imfill(BW,'holes');
figure;
imshow(filled);



%%%%%   the indexes for the images are counted up and down.
%%%%%%%  i mean to say that in the diagram .. index of 2 is 1 ,, 

%%%% index of 5 is 3 ,, index of 8 is 4,, index of 3 is 2,, 
%%%%% index of 7 is 5...so on