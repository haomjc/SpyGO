function [pt,ptn]=point1
ml=5;
nl=10;
%划分网格
global r tn1 tn2 gamma adg deg rc2 sr2 q2 em2 xb2 xg2 ...
    b2 cr2 alp2 ad2 af2 fw2 aa2 pgma2
%基本参数
tn1=38; tn2=51; gamma=90; adg=2.57070; deg=4.60370;
rc2=96.265; sr2=95.559965; q2= -54.735452;
em2=0; xb2=-0.000004; xg2=0; b2=35;
cr2=0.802465; alp2=-18.000000; ad2=2.181757;
af2=109.840600 ; fw2=22.000000 ; aa2=1.557223;

%转化为弧度
d2h=@(x) x*pi/180;
alp2=d2h(alp2); aa2=d2h(aa2); b2=d2h(b2);
gamma=d2h(gamma); q2=d2h(q2); ad2=d2h(ad2);
%计算节锥角
mm21=tn1/tn2;
pgma2=atan(sin(gamma)/(mm21+cos(gamma)));
if(pgma2<0.0)
    pgma2=pgma2+pi;
end
%向S2m转换的矩阵
snpit2=sin(pgma2);
cspit2=cos(pgma2);
Mp2s=[cspit2 -snpit2
    snpit2 cspit2 ];
%取四角点
afi2=af2-fw2/2;
afo2=af2+fw2/2;
gx(1)=afi2; gx(2)=afo2;
gy(1)=-deg+fw2*tan(ad2);
gy(2)=-deg;
gx(3)=afo2; gx(4)=afi2;
gy(3)=adg; gy(4)=adg-fw2*tan(aa2);
%找M点
r(1)=af2; r(2)=0; r=[r(1); r(2)];
r=Mp2s*r;
options=[0,1e-9,1e-9];
%options=['Display','iter'];
%x0=[0; 0.001];
x0(1)=d2h(270)+1.57-b2; x0(2)=0;
x0=fsolve(@(x) gearface(x),x0,options);
%取结点ml*nl
for i=1:ml
    x1=x0;
    for j=1:nl
        p(i,j,1)=gx(1)+(j-1)*(gx(2)-gx(1))/(nl-1);

        y1=gy(4)-(i-1)*(gy(4)-gy(1))/(ml-1);
        y2=gy(3)-(i-1)*(gy(3)-gy(2))/(ml-1);
        p(i,j,2)=y1+(p(i,j,1)-gx(1))*(y2-y1)/(gx(3)-gx(4));
        %放入S2m中
        r=[p(i,j,1); p(i,j,2)]; r=Mp2s*r;
        %求齿面坐标..................
        %二维
        %初值

        %        x0=[0.0;0.001];
        %精度控制
        options=[0,1e-9,1e-9];
        % options=['Display','iter'];
        %求解
        x=fsolve(@(x) gearface(x),x1,options);
        clc %清屏
        x1=x;
        %三维
        [y,R,N]=gearface(x);
        for k=1:3
            pt(i,j,k)=R(k);
            ptn(i,j,k)=N(k);
        end
    end
end
%surf(pt(:,:,1),pt(:,:,2),pt(:,:,3));
%******************************************
function [y,R2,n2]=gearface(x)
    %基本参数
    global r tn1 tn2 gamma adg deg rc2 sr2 q2 em2 xb2 xg2 ...
        b2 cr2 alp2 ad2 af2 fw2 aa2 pgma2
    th=x(1); ph=x(2);
    %计算根锥角
    gama2=pgma2-ad2;
    %三角函数
    sp=sin(alp2); cp=cos(alp2);
    sm=sin(gama2); cm=cos(gama2);
    phi2=ph/cr2;  %轮坯转角
    sh2=sin(phi2);  ch2=cos(phi2);
    stp=sin(th-ph); ctp=cos(th-ph);
    %表示sg
    nm=[-cp*ctp; -cp*stp; sp]; %锥面在Sm2的法矢
    a=[-em2*sm; xb2*cm; em2*cm];
    aa1=rc2*stp+sr2*sin(-q2-ph);
    aa2=rc2*ctp+sr2*cos(-q2-ph);
    t1=[-aa1*(sm-cr2); aa2*(sm-cr2); aa1*cm];
    t2=[-(sm-cr2)*sp*stp
        (sm-cr2)*sp*ctp-cp*cm
        cm*sp*stp ];
    sg=dot(nm,(a+t1))/dot(nm,t2);
    %大轮齿面的二参数方程
    %锥面在Sm2的方程
    Rm=[(rc2-sg*sp)*ctp+sr2*cos(-q2-ph)
        (rc2-sg*sp)*stp+sr2*sin(-q2-ph)
        -sg*cp
        1];
    %转换阵
    %m2-->d2
    Ld2m2=[cm 0 sm; 0 1 0; -sm 0 cm]; lj=[-xb2*sm-xg2; em2; -xb2*cm; 1];
    Md2m2=zeros(4,4); Md2m2(1:3,1:3)=Ld2m2; Md2m2(:,4)=lj;
    %d2-->S2
    Ls2d2=[1 0 0; 0 ch2 -sh2; 0 sh2 ch2]; lj=[0 0 0 1];
    Ms2d2=zeros(4,4); Ms2d2(1:3,1:3)=Ls2d2; Ms2d2(:,4)=lj;
    %计算大轮的齿面方程和法矢
    R2=Ms2d2*Md2m2*Rm;
    n2=Ls2d2*Ld2m2*nm;
    %需解方程式
    y(1)=r(1)-R2(1);
    y(2)=R2(2)^2+R2(3)^2-r(2)^2;
    y=[y(1) y(2)];