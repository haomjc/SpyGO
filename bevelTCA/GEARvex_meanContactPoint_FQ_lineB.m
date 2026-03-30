%---------确定在计算参考点处满足传动比等于齿数比的安装调整参数值的fsolve子函数   GEARcvex_meanContactPoint_FQ_lineB.m-------%

function Fmp2=GEARvex_meanContactPoint_FQ_lineB(x,thetaG,phic2,thetaP,phic1)

thetaX=x(1);
thetaY=x(2);
thetaZ=x(3);
dxk2=x(4);
dyk2=x(5);
dzk2=x(6);

global ZP ZG du2 PW Sr2 q2 XB2 XD2 Em2 mcG gamaf2 alphaG dc1 Sr1 q1 XB1 XD1 Em1 mcP1 gamaf1 alphaP C D

rc2=(du2-PW)/2;   %……加工凸面时的刀尖半径

phi2=mcG*phic2;   %……加工时大轮的转角

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

rc1=dc1/2;   %……加工凹面时的刀尖半径(小轮)

mcP=1/(mcP1*(1-2*C*phic1-3*D*phic1^2));%……切削滚比(小轮)
phi1=mcP1*(phic1-C*phic1^2-D*phic1^3); %……加工时小轮的转角

sP=Sr1*(mcP-sin(gamaf1))*cos(alphaP)*sin(thetaP-q1)/(cos(gamaf1)*sin(phic1+thetaP))-(Sr1*sin(phic1+q1)+Em1)*sin(alphaP)/sin(phic1...
    +thetaP)+Em1*tan(gamaf1)*cos(alphaP)/tan(phic1+thetaP)-rc1*sin(alphaP)-XB1*cos(alphaP);

%………………由刀盘坐标系Sb1到固连于被加工大轮坐标系S1的各坐标变换矩阵………………%
Mc1P=[1 0 0 Sr1*cos(q1);0 1 0 Sr1*sin(q1);0 0 1 0;0 0 0 1];                          %……SP-->Sc1
Mm1c1=[cos(phic1) -sin(phic1) 0 0;sin(phic1) cos(phic1) 0 0;0 0 1 0;0 0 0 1];        %……Sc1-->Sm1
Ma1m1=[1 0 0 0;0 1 0 Em1;0 0 1 -XB1;0 0 0 1];                                        %……Sm1-->Sa1
Mb1a1=[sin(gamaf1) 0 -cos(gamaf1) 0;0 1 0 0;cos(gamaf1) 0 sin(gamaf1) -XD1;0 0 0 1]; %……Sa1-->Sb1
M1b1=[cos(phi1) sin(phi1) 0 0;-sin(phi1) cos(phi1) 0 0;0 0 1 0;0 0 0 1];             %……Sb1-->S1

Lm1c1=[cos(phic1) -sin(phic1) 0;sin(phic1) cos(phic1) 0;0 0 1];
Lb1a1=[sin(gamaf1) 0 -cos(gamaf1);0 1 0;cos(gamaf1) 0 sin(gamaf1)];
L1b1=[cos(phi1) sin(phi1) 0;-sin(phi1) cos(phi1) 0;0 0 1];

%………………加工凹面的刀具切削刃圆锥面方程及其法向量在Sb1中的表达
rP=[(rc1+sP*sin(alphaP))*cos(thetaP);(rc1+sP*sin(alphaP))*sin(thetaP);-sP*cos(alphaP);1];
nP=[cos(alphaP)*cos(thetaP);cos(alphaP)*sin(thetaP);sin(alphaP)];

%………………加工凹面的刀具切削刃圆锥面方程在S1中的表达
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

nm2=Lma*Lab*Lbc*Lch*n2;
rm2=Mtr*Mma*Mab*Mbc*Mch*r2;
%——————经过上述变换过程，大小轮齿面计算基点重合，基点处法矢共线——————%

%………………设小、大齿轮的角速度分别为：w1,w2
w1=[0;0;1];
w2=[0;0;-ZP/ZG];
wm2=Lma*Lab*Lbc*Lch*w2;

%………………大小轮的相对角速度
w12=w1-wm2;

o2o1=-((Lma*Lab*Lbc*Lch)'*[dxk2;dyk2;dzk2]);

%………………求大小齿轮的相对速度
v0=[w12(2)*r1(3)-r1(2)*w12(3);r1(1)*w12(3)-w12(1)*r1(3);w12(1)*r1(2)-r1(1)*w12(2)];
v1=[wm2(2)*o2o1(3)+o2o1(2)*wm2(3);o2o1(1)*wm2(3)+wm2(1)*o2o1(3);wm2(1)*o2o1(2)+o2o1(1)*wm2(2)];
v12=v0-v1;

%………………确定初始安装位置的基本方程组构造………………%
Fmp2(1)=r1(1)-rm2(1);
Fmp2(2)=r1(2)-rm2(2);
Fmp2(3)=r1(3)-rm2(3);
Fmp2(4)=n1(1)+nm2(1);
Fmp2(5)=n1(3)+nm2(3);
Fmp2(6)=nm2(1)*v12(1)+nm2(2)*v12(2)+nm2(3)*v12(3);
