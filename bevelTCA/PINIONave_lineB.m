
%--------求解小齿轮凹面的fsolve子函数  PINIONave_lineB.m

function FP=PINIONave_parabolaB(x,LP,RP)

thetaP=x(1);
phic1=x(2);

global ha1 hf1 dc1 Sr1 q1 XB1 XD1 Em1 mcP1 gamaf1 alphaP C D A10 aP

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

%………………加工凹面的刀具切削刃圆锥面方程及其法向量在Sb1中的表达
rP=[(rc1+sP*sin(alphaP))*cos(thetaP);(rc1+sP*sin(alphaP))*sin(thetaP);-sP*cos(alphaP);1];

%………………加工凹面的刀具切削刃圆锥面方程在S1中的表达
r1=M1b1*Mb1a1*Ma1m1*Mm1c1*Mc1P*rP;

%………………齿面点(凹面)在三维直角坐标系中各坐标分量的表达式
x1=r1(1);
y1=r1(2);
z1=r1(3);

%………………构造非线性方程组
FP(1)=z1-LP;
FP(2)=sqrt(x1^2+y1^2)-RP;
