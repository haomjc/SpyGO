clear
clc
clf
hold on
d2=193.80000;
[pt,ptn]=point1;
pt1=pt;
ptn1=ptn;
[pt,ptn]=point2;
pt2=pt;
ptn2=ptn;
surf(pt1(:,:,1),pt1(:,:,2),pt1(:,:,3));
[a1,b1,c1]=size(pt);
p1=polyfit(pt1(:,b1,2),pt1(:,b1,3),3);
p2=polyfit(pt2(:,b1,2),pt2(:,b1,3),3);
syms y
z1=p1(1)*y^3+p1(2)*y^2+p1(3)*y+p1(4);
z2=p2(1)*y^3+p2(2)*y^2+p2(3)*y+p2(4);
y1=solve(y^2+z1^2-(d2/2)^2);
y2=solve(y^2+z2^2-(d2/2)^2);
y1=double(y1);
y2=double(y2);
Y1=y1(1);
Y2=y2(1);
Z1=subs(z1,Y1);
Z2=subs(z2,Y2);
X1=interp1(pt1(:,b1,2),pt1(:,b1,1),Y1,'spline');
X2=interp1(pt2(:,b1,2),pt2(:,b1,1),Y2,'spline');
A=5.25394/d2*2+2*atan(sqrt((X2-X1)^2+(Y2-Y1)^2)/d2);
for i=1:a1
    pt2q(:,:,i)=[pt2(i,:,1)',pt2(i,:,2)',pt2(i,:,3)',ones(b1,1)];
    pt3q(:,:,i)=pt2q(:,:,i)*[1 0 0 0;0 cos(A) sin(A) 0;0 -sin(A) cos(A) 0;0 0 0 1];
end
for i=1:a1
    pt3(i,:,1)=pt3q(:,1,i)';
    pt3(i,:,2)=pt3q(:,2,i)';
    pt3(i,:,3)=pt3q(:,3,i)';
end
surf(pt3(:,:,1),pt3(:,:,2),pt3(:,:,3));
X=[pt1(1,:,1);pt3(1,:,1)];
Y=[pt1(1,:,2);pt3(1,:,2)];
Z=[pt1(1,:,3);pt3(1,:,3)];
surf(X,Y,Z);
for i=1:a1
    pt1q(:,:,i)=[pt1(a1-i+1,:,1)',pt1(a1-i+1,:,2)',pt1(a1-i+1,:,3)',ones(b1,1)];
end
surf([pt1(:,1,1)';pt3(:,1,1)'],[pt1(:,1,2)';pt3(:,1,2)'],[pt1(:,1,3)';pt3(:,1,3)']);
surf([pt1(:,b1,1)';pt3(:,b1,1)'],[pt1(:,b1,2)';pt3(:,b1,2)'],[pt1(:,b1,3)';pt3(:,b1,3)']);
for i=1:50
    AA=2*pi/51*i;
    for j=1:a1
        pt1h(:,:,j)=pt1q(:,:,j)*[1 0 0 0;0 cos(AA) sin(AA) 0;0 -sin(AA) cos(AA) 0;0 0 0 1];
        pt2h(:,:,j)=pt3q(:,:,j)*[1 0 0 0;0 cos(AA) sin(AA) 0;0 -sin(AA) cos(AA) 0;0 0 0 1];
        PT1(j,:,1)=pt1h(:,1,j)';
        PT1(j,:,2)=pt1h(:,2,j)';
        PT1(j,:,3)=pt1h(:,3,j)';
        PT2(j,:,1)=pt2h(:,1,j)';
        PT2(j,:,2)=pt2h(:,2,j)';
        PT2(j,:,3)=pt2h(:,3,j)';
    end
    surf([PT1(:,:,1);PT2(:,:,1)],[PT1(:,:,2);PT2(:,:,2)],[PT1(:,:,3);PT2(:,:,3)]);
    surf([flipdim(PT1(:,1,1)',2);PT2(:,1,1)'],[flipdim(PT1(:,1,2)',2);PT2(:,1,2)'],[flipdim(PT1(:,1,3)',2);PT2(:,1,3)']);
    surf([flipdim(PT1(:,b1,1)',2);PT2(:,b1,1)'],[flipdim(PT1(:,b1,2)',2);PT2(:,b1,2)'],[flipdim(PT1(:,b1,3)',2);PT2(:,b1,3)']);
end
h1=sqrt(pt(a1,1,2)^2+pt(a1,1,3)^2);
h2=sqrt(pt(a1,b1,2)^2+pt(a1,b1,3)^2);
for i=1:10
    r=h1+(h2-h1)/9*(i-1);
    t=0:pi/20:2*pi;
    yy(i,:)=r*sin(t);
    zz(i,:)=r*cos(t);
end
x=linspace(pt(a1,1,1),pt(a1,b1,1),10);
for i=1:10
    xx(i,:)=ones(1,length(yy))*x(i);
end
surf(xx,yy,zz);
clear
axis equal
axis vis3d
view(270,0)
hold off