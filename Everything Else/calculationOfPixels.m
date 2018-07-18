function [ width,height ] = calculationOfPixels( imagefile )
%width and height are variables calculated by this function 
%we take image file as input
%round the dimensions so that the product is 784

I=imread(imagefile);
[h,w]=size(I);

h1=h;h2=h;w1=w;w2=w;
while mod(h1,28)~=0 ||  mod(h2,28)~=0
    h1=h1-1;
    h2=h2+1;
end

if  mod(h1,28)==0
    h=h1;
else
    h=h2;
end

while  mod(w1,28)~=0 ||  mod(w2,28)~=0
    w1=w1-1;
    w2=w2+1;
end

if  mod(w1,28)==0
    w=w1;
else
    w=w2;
end

h1=h;w1=w;
if h1>w1
    while w1*h1~=784
        h1=h1-7;
    end

else
    while w1*h1~=784
        w1=w1-7;
    end
end

w=w1;h=h1;

width=w;height=h;
end

