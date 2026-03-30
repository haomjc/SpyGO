%---------确定满足安装位置的thetaG、phic2、thetaP、phic1、pc2参数的值的fsolve子函数   PRHave_GLHvex_meshing_lineB.m

function Fm2=PRHave_GLHvex_meshing_lineB(x,pc1,thetaX,thetaY,thetaZ,dxk2,dyk2,dzk2)

thetaG=x(1);
phic2=x(2);
thetaP=x(3);
phic1=x(4);
pc2=x(5);

global du2 PW Sr2 q2 XB2 XD2 Em2 mcG gamaf2 alphaG dc1 Sr1 q1 XB1 XD1 Em1 mcP1 gamaf1 alphaP C D eAX eOS eT

rc2=(du2-PW)/2;  %……加工凸面时的刀尖半径

phi2=mcG*phic2;  %……加工时大轮的转角

sG=Sr2*(1/mcG-sin(gamaf2))*cos(alphaG)*sin(thetaG-q2)/(cos(gamaf2)*sin(phic2+thetaG))+(Sr2*sin(phic2+q2)+Em2)*sin(alphaG)...
    /sin(phic2+thetaG)-Em2*tan(gamaf2)*cos(alphaG)/tan(phic2+thetaG)+rc2*sin(alphaG)-XB2*cos(alphaG);

%………………由刀盘坐标系Sb2到固连于被加工大轮坐标系S2的各坐标变换矩阵………………%
Mc2G=[1 0 0 Sr2*cos(q2);0 1 0 Sr2*sin(q2);0 0 1 0;0 0 0 1];                          %……SG-->Sc2
Mm2c2=[cos(phic2) -sin(phic2) 0 0;sin(phic2) cos(phic2) 0 0;0 0 1 0;0 0 0 1];        %……Sc2-->Sm2
Ma2m2=[1 0 0 0;0 1 0 Em2;0 0 1 -XB2;0 0 0 1];                                        %……Sm2-->Sa2
Mb2a2=[sin(gamaf2) 0 -cos(gamaf2) 0;0 1 0 0;cos(gamaf2) 0 sin(gamaf2) -XD2;0 0 0 1]; %……Sa2-->Sb2
M2b2=[cos(phi2) sin(phi2) 0 0;-sin(phi2) cos(phi2) 0 0;0 0 1 0;0 0 0 1];             %……Sb2-->S2

%………………上面对应矩阵的子阵
Lm2c2=[cos(phic2) -sin(phic2) 0;sin(phic2) cos(phic2) 0;0 0 1];
Lb2a2=[sin(gamaf2) 0 -cos(gamaf2);0 1 0;cos(gamaf2) 0 sin(gamaf2)];
L2b2=[cos(phi2) sin(phi2) 0;-sin(phi2) cos(phi2) 0;0 0 1];

%………………加工凸面的刀具切削刃圆锥面方程及其法向量在Sb中的表达
rG=[(rc2-sG*sin(alphaG))*cos(thetaG);(rc2-sG*sin(alphaG))*sin(thetaG);-sG*cos(alphaG);1];
nG=[cos(alphaG)*cos(thetaG);cos(alphaG)*sin(thetaG);-sin(alphaG)];

%………………加工凸面的刀具切削刃圆锥面方程在S2中的表达
r2=M2b2*Mb2a2*Ma2m2*Mm2c2*Mc2G*rG;
n2=L2b2*Lb2a2*Lm2c2*nG;

rc1=dc1/2;   %……加工凹面时的刀尖半径

mcP=1/(mcP1*(1-2*C*phic1-3*D*phic1^2));%……切削滚比
phi1=mcP1*(phic1-C*phic1^2-D*phic1^3); %……加工时小轮的转角

sP=Sr1*(mcP-sin(gamaf1))*cos(alphaP)*sin(thetaP-q1)/(cos(gamaf1)*sin(phic1+thetaP))-(Sr1*sin(phic1+q1)+Em1)*sin(alphaP)/sin(phic1...
    +thetaP)+Em1*tan(gamaf1)*cos(alphaP)/tan(phic1+thetaP)-rc1*sin(alphaP)-XB1*cos(alphaP);

%………………由刀盘坐标系Sb1到固连于被加工大轮坐标系S1的各坐标变换矩阵………………%
Mc1P=[1 0 0 Sr1*cos(q1);0 1 0 Sr1*sin(q1);0 0 1 0;0 0 0 1];                          %……SP-->Sc1
Mm1c1=[cos(phic1) -sin(phic1) 0 0;sin(phic1) cos(phic1) 0 0;0 0 1 0;0 0 0 1];        %……Sc1-->Sm1
Ma1m1=[1 0 0 0;0 1 0 Em1;0 0 1 -XB1;0 0 0 1];                                        %……Sm1-->Sa1
Mb1a1=[sin(gamaf1) 0 -cos(gamaf1) 0;0 1 0 0;cos(gamaf1) 0 sin(gamaf1) -XD1;0 0 0 1]; %……Sa1-->Sb1
M1b1=[cos(phi1) sin(phi1) 0 0;-sin(phi1) cos(phi1) 0 0;0 0 1 0;0 0 0 1];             %……Sb1-->S1

%………………上面对应矩阵的子阵
Lm1c1=[cos(phic1) -sin(phic1) 0;sin(phic1) cos(phic1) 0;0 0 1];
Lb1a1=[sin(gamaf1) 0 -cos(gamaf1);0 1 0;cos(gamaf1) 0 sin(gamaf1)];
L1b1=[cos(phi1) sin(phi1) 0;-sin(phi1) cos(phi1) 0;0 0 1];

%………………加工凹面的刀具切削刃圆锥面方程及其法向量在Sb1中的表达
rP=[(rc1+sP*sin(alphaP))*cos(thetaP);(rc1+sP*sin(alphaP))*sin(thetaP);-sP*cos(alphaP);1];
nP=[cos(alphaP)*cos(thetaP);cos(alphaP)*sin(thetaP);sin(alphaP)];

%………………加工凸面的刀具切削刃圆锥面方程在S1中的表达
r1=M1b1*Mb1a1*Ma1m1*Mm1c1*Mc1P*rP;
n1=L1b1*Lb1a1*Lm1c1*nP;

%………………法向量共线n2的变换过程(revolution)
Lch=[0 0 -1;0 1 0;1 0 0];
Lbc=[1 0 0;0 cos(thetaX) -sin(thetaX);0 sin(thetaX) cos(thetaX)];%………
Lab=[cos(thetaY) 0 sin(thetaY);0 1 0;-sin(thetaY) 0 cos(thetaY)];%………
Lma=[cos(thetaZ) -sin(thetaZ) 0;sin(thetaZ) cos(thetaZ) 0;0 0 1];%………

%………………基点重合r2的变换过程(revolution and translation)
Mch=[0 0 -1 0;0 1 0 0;1 0 0 0;0 0 0 1];%………
Mbc=[1 0 0 0;0 cos(thetaX) -sin(thetaX) 0;0 sin(thetaX) cos(thetaX) 0;0 0 0 1];%………
Mab=[cos(thetaY) 0 sin(thetaY) 0;0 1 0 0;-sin(thetaY) 0 cos(thetaY) 0;0 0 0 1];%………
Mma=[cos(thetaZ) -sin(thetaZ) 0 0;sin(thetaZ) cos(thetaZ) 0 0;0 0 1 0;0 0 0 1];%………
Mtr=[1 0 0 -dxk2;0 1 0 -dyk2;0 0 1 -dzk2;0 0 0 1];%………
%………………齿轮副啮合坐标系的变换矩阵
Mm1=[cos(pc1) -sin(pc1) 0 0;sin(pc1) cos(pc1) 0 0;0 0 1 0;0 0 0 1];%………………S1-->Sm
Mh2=[cos(pc2) -sin(pc2) 0 0;sin(pc2) cos(pc2) 0 0;0 0 1 0;0 0 0 1];%………………S2-->Sh

%………………上面对应矩阵的子阵
Lm1=[cos(pc1) -sin(pc1) 0;sin(pc1) cos(pc1) 0;0 0 1];
Lh2=[cos(pc2) -sin(pc2) 0;sin(pc2) cos(pc2) 0;0 0 1];

nm2=Lma*Lab*Lbc*Lch*Lh2*n2;
rm2=Mtr*Mma*Mab*Mbc*Mch*Mh2*r2;

%………………安装误差存在时的坐标变换矩阵
Mij=[1 0 0 0;0 1 0 0;0 0 1 eAX;0 0 0 1];
Mjk=[1 0 0 0;0 1 0 -eOS;0 0 1 0;0 0 0 1];
Mmi=[cos(eT) 0 sin(eT) 0;0 1 0 0;-sin(eT) 0 cos(eT) 0;0 0 0 1];

Ljk=[1 0 0;0 1 0;0 0 1];
Lij=[1 0 0;0 1 0;0 0 1];
Lmi=[cos(eT) 0 sin(eT);0 1 0;-sin(eT) 0 cos(eT)];

%………………在啮合坐标系Sm中大轮齿面坐标的分量表达式
rm1=Mmi*Mij*Mjk*Mm1*r1;
nm1=Lmi*Lij*Ljk*Lm1*n1;

%………………ＴＣＡ基本方程组的构造………………%
Fm2(1)=rm2(1)-rm1(1);
Fm2(2)=rm2(2)-rm1(2);
Fm2(3)=rm2(3)-rm1(3);
Fm2(4)=nm2(3)+nm1(3);
Fm2(5)=nm2(2)+nm1(2);
