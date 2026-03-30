
%--------求解大齿轮凸面的fsolve子函数  GEARvex_lineB.m--------%

function FG=GEARvex_lineB(x,LG,RG)

thetaG=x(1);
phic2=x(2);

global ha2 hf2 du2 PW Sr2 q2 XB2 XD2 Em2 mcG gamaf2 alphaG A0 aG

rc2=(du2-PW)/2;  %……加工凸面时的刀尖半径

phi2=mcG*phic2;  %……加工时大轮的转角

%………………由刀盘坐标系Sb2到固连于被加工大轮坐标系S2的各坐标变换矩阵………………%
Mc2G=[1 0 0 Sr2*cos(q2);0 1 0 Sr2*sin(q2);0 0 1 0;0 0 0 1];                          %……SG-->Sc2
Mm2c2=[cos(phic2) -sin(phic2) 0 0;sin(phic2) cos(phic2) 0 0;0 0 1 0;0 0 0 1];        %……Sc2-->Sm2
Ma2m2=[1 0 0 0;0 1 0 Em2;0 0 1 -XB2;0 0 0 1];                                        %……Sm2-->Sa2
Mb2a2=[sin(gamaf2) 0 -cos(gamaf2) 0;0 1 0 0;cos(gamaf2) 0 sin(gamaf2) -XD2;0 0 0 1]; %……Sa2-->Sb2
M2b2=[cos(phi2) sin(phi2) 0 0;-sin(phi2) cos(phi2) 0 0;0 0 1 0;0 0 0 1];             %……Sb2-->S2

sG=Sr2*(1/mcG-sin(gamaf2))*cos(alphaG)*sin(thetaG-q2)/(cos(gamaf2)*sin(phic2+thetaG))+(Sr2*sin(phic2+q2)+Em2)*sin(alphaG)...
    /sin(phic2+thetaG)-Em2*tan(gamaf2)*cos(alphaG)/tan(phic2+thetaG)+rc2*sin(alphaG)-XB2*cos(alphaG);

%………………加工凸面的刀具切削刃圆锥面方程及其法向量在Sb中的表达
rG=[(rc2-sG*sin(alphaG))*cos(thetaG);(rc2-sG*sin(alphaG))*sin(thetaG);-sG*cos(alphaG);1];

%………………加工凸面的刀具切削刃圆锥面方程在S2中的表达
r2=M2b2*Mb2a2*Ma2m2*Mm2c2*Mc2G*rG;

%………………齿面点在三维直角坐标系中各坐标分量的表达式
x2=r2(1);
y2=r2(2);
z2=r2(3);

%………………构造非线性方程组
FG(1)=z2-LG;
FG(2)=sqrt(x2^2+y2^2)-RG;
