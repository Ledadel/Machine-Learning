%This was a program I created in order to test the speed matrix
%multiplication with matlab vs. octave.  I created the matrices sperately
%so that the time to create the random matrices is not included in the
%calculation
clear all;
load('randmats');
tic
a=.001*a;
b=.001*b;
c=a*b;
d=ones(1,length(c))*c*ones(length(c),1)
toc