% This M-file constructs the individual images for 60 digits
% and plots them to a file.

clear
format short g
load zip.train
digits=zip(:,1);
grayscale=zip(:,2:end);

[n,d]=size(grayscale);
w=floor(sqrt(d));

for i=1:100
	[i, digits(i)]
	curimage=reshape(grayscale(i,:),w,w);
	curimage=curimage';
	l=displayimage(curimage);
	sstr=['IndividualImages/image',int2str(i)];
%	eval(['print -deps ',sstr]);
end
