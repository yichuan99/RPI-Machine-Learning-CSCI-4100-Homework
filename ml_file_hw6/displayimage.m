function out=displayimage(curimage)
%This is a functions that creates a graphical image of a 
%of a digit which is a w x w grayscale matrix.

[m,n]=size(curimage);
%implus=(curimage<-0.1);
im=zeros(m,n,3);
for i=1:3	
	im(:,:,i)=0.5*(1-curimage);
%	im(:,:,i)=implus;
end

out=image(im);
h=gca;
set(h,'XTick',[]);
set(h,'YTick',[]);
