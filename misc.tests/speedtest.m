%This was a program I created in order to test the speed matrix
%multiplication with matlab vs. octave.  I created the matrices sperately
%so that the time to create the random matrices is not included in the
%calculation
clear all;
%load('randmats');
tic
a=rand(1000);
b=rand(1000);
toc
a=.001*a;
b=.001*b;
tic
c=a*b;
toc
d=ones(1,length(c))*c*ones(length(c),1) %#ok<NOPTS>
