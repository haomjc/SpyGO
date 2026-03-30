
%---------------大轮凸面与小轮凹面的TCA       liu_PINIONave_GEARvex_meshing_lineB.m---------------%

%编程：刘万春        修改日期：2007－12－24

%程序功能结构：确定齿面投影图－－>确定齿轮副初始安装位置调整参数－－>计算齿面啮合迹和传动误差曲线－－>图形显示

clear all
tic;       %………………开始计时语句，与最后的toc配合

global du2 PW Sr2 q2 XB2 XD2 Em2 mcG gamaf2 alphaG dc1 Sr1 q1 XB1 XD1 Em1 mcP1 gamaf1 alphaP C D ZP ZG eAX eOS eT

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>大、小轮基本参数，加工参数（刀具参数、机床调整参数）输入<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%

load BlankData_input.txt;  %………………打开基本参数的txt输入文件

ZP=BlankData_input(1);
ZG=BlankData_input(2);
mn=BlankData_input(3);
La=BlankData_input(4);
B=BlankData_input(5);

%………………大轮基本参数………………%
ha2=BlankData_input(6);
hf2=BlankData_input(7);

gama20=BlankData_input(8);   %………………大轮节锥角
gama2=gama20*pi/180;
gamaf20=BlankData_input(9);  %………………大轮根锥角
gamaf2=gamaf20*pi/180;
gamaa20=BlankData_input(10);  %………………大轮面锥角
gamaa2=gamaa20*pi/180;

dertaa2=gamaa2-gama2;%………………大轮齿顶角
dertaf2=gama2-gamaf2;%………………大轮齿根角

%………………小轮基本参数………………%
ha1=BlankData_input(11);
hf1=BlankData_input(12);

gama10=BlankData_input(13);  %………………小轮节锥角
gama1=gama10*pi/180;
gamaf10=BlankData_input(14); %………………小轮根锥角
gamaf1=gamaf10*pi/180;
gamaa10=BlankData_input(15); %………………小轮面锥角
gamaa1=gamaa10*pi/180;

dertaa1=dertaf2; %………………小轮齿顶角
dertaf1=dertaa2; %………………小轮齿根角

qVH=BlankData_input(19);%………………V-H检验调整系数

load GearMachiningParameter_input.txt;   %………………打开大轮加工调整参数的txt输入文件

alphaG0=GearMachiningParameter_input(2); %………………加工大轮凸面时的刀具齿形角
alphaG=alphaG0*pi/180;

%………………加工大轮齿面的刀具参数和加工调整参数………………%
du2=GearMachiningParameter_input(3);       %………………刀盘公称直径
PW=GearMachiningParameter_input(4);        %………………刀顶距
Sr2=GearMachiningParameter_input(5);       %………………径向刀位
q20=GearMachiningParameter_input(6);       %………………角向刀位
q2=q20*pi/180;
XB2=GearMachiningParameter_input(7);       %………………床位
XD2=GearMachiningParameter_input(8);       %………………轴向轮位
Em2=GearMachiningParameter_input(9);       %………………垂直轮位
mcG=GearMachiningParameter_input(10);      %………………切削滚比
rog=GearMachiningParameter_input(13);      %………………加工大轮的刀尖过度圆角半径

load PinionMachiningParameter_input.txt;   %………………打开小轮加工调整参数的txt输入文件

C=PinionMachiningParameter_input(3);
D=PinionMachiningParameter_input(4);      %……切削滚比修正系数（小轮凹面）

alphaP0=PinionMachiningParameter_input(6); %………………加工小轮凹面时的刀具齿形角
alphaP=alphaP0*pi/180;

%………………加工小轮齿面（凹）的刀具参数和加工调整参数………………%
rop=PinionMachiningParameter_input(8);    %………………加工小轮的刀尖过度圆角半径
dc1=PinionMachiningParameter_input(15);   %………………刀尖直径（凹）
Sr1=PinionMachiningParameter_input(16);   %………………径向刀位
q10=PinionMachiningParameter_input(17);   %………………角向刀位
q1=q10*pi/180;
XB1=PinionMachiningParameter_input(18);   %………………床位
XD1=PinionMachiningParameter_input(19);   %………………轴向轮位
Em1=PinionMachiningParameter_input(20);   %………………垂直轮位
mcP1=PinionMachiningParameter_input(21);  %………………切削滚比

%………………安装误差………………%
eAX=PinionMachiningParameter_input(26);   %………………齿圈轴向位移偏差
eOS=PinionMachiningParameter_input(27);   %………………齿轮副轴间距偏差
eT=PinionMachiningParameter_input(28);    %………………齿轮副轴交角偏差

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>确定大、小轮齿面投影<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%

%………………大轮齿面投影网格节点的确定………………%

%………………确定计算参考点投影位置
hf20=hf2-rog*(1-sin(alphaG));
c2=0.188*mn;
hfG=hf2-qVH*B*tan(dertaf2);
haG=ha2-qVH*B*tan(dertaa2);
hm2=haG+hfG;
Pm2=La-qVH*B;

LGmpr=Pm2*cos(gama2)+(hfG-(hm2+c2)/2)*sin(gama2);
RGmpr=Pm2*sin(gama2)-(hfG-(hm2+c2)/2)*cos(gama2);

%………………计算有效齿面投影网格四个角点(1,1)、(m,1)、(1,n)、(m,n)的L和R值，“RG”中的“G”代表大齿轮
LG11=(La-B)*cos(gama2)-(ha2-B*tan(dertaa2))*sin(gama2);
RG11=(La-B)*sin(gama2)+(ha2-B*tan(dertaa2))*cos(gama2);

LGm1=La*cos(gama2)-ha2*sin(gama2);
RGm1=La*sin(gama2)+ha2*cos(gama2);

LG1nr=(La-B)*cos(gama2)+(hf20-B*tan(dertaf2))*sin(gama2);
RG1nr=(La-B)*sin(gama2)-(hf20-B*tan(dertaf2))*cos(gama2);

LGmnr=La*cos(gama2)+hf20*sin(gama2);
RGmnr=La*sin(gama2)-hf20*cos(gama2);

%………………计算齿根投影网格角点(1,n0)、(m,n0)的L和R值
LG1n0=(La-B)*cos(gama2)+(hf2-B*tan(dertaf2))*sin(gama2);
RG1n0=(La-B)*sin(gama2)-(hf2-B*tan(dertaf2))*cos(gama2);

LGmn0=La*cos(gama2)+hf2*sin(gama2);
RGmn0=La*sin(gama2)-hf2*cos(gama2);

%………………小轮齿面投影网格节点的确定………………%

%………………确定计算参考点投影位置
hf10=hf1-rop*(1-sin(alphaP));
hfP=hf1-qVH*B*tan(dertaf1);
haP=ha1-qVH*B*tan(dertaa1);
hm1=haP+hfP;
Pm1=La-qVH*B;
LPmpr=Pm1*cos(gama1)+(hfP-(hm1+c2)/2)*sin(gama1);
RPmpr=Pm1*sin(gama1)-(hfP-(hm1+c2)/2)*cos(gama1);

%………………计算有效齿面投影网格四个角点(1,1)、(m,1)、(1,n)、(m,n)的L和R值，“RP”中的“P”代表小齿轮
LP11=(La-B)*cos(gama1)-(ha1-B*tan(dertaa1))*sin(gama1);
RP11=(La-B)*sin(gama1)+(ha1-B*tan(dertaa1))*cos(gama1);

LPm1=La*cos(gama1)-ha1*sin(gama1);
RPm1=La*sin(gama1)+ha1*cos(gama1);

LP1nr=(La-B)*cos(gama1)+(hf10-B*tan(dertaf1))*sin(gama1);
RP1nr=(La-B)*sin(gama1)-(hf10-B*tan(dertaf1))*cos(gama1);

LPmnr=La*cos(gama1)+hf10*sin(gama1);
RPmnr=La*sin(gama1)-hf10*cos(gama1);

%………………计算齿根投影网格节点(1,n0)、(m,n0)的L和R值
LP1n0=(La-B)*cos(gama1)+(hf1-B*tan(dertaf1))*sin(gama1);
RP1n0=(La-B)*sin(gama1)-(hf1-B*tan(dertaf1))*cos(gama1);

LPmn0=La*cos(gama1)+hf1*sin(gama1);
RPmn0=La*sin(gama1)-hf1*cos(gama1);

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>大轮齿面点坐标（x2,y2,z2)，小轮齿面点坐标（x1,y1,z1)描述<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%

%………………大轮齿面(凸面)点坐标（x2,y2,z2)在S2中的描述………………%
rc2=(du2-PW)/2;   %……加工凸面时的刀尖半径

syms thetaG phic2 %……定义参数为计算符号

phi2=mcG*phic2;   %……加工时大轮的转角

%………………由刀盘坐标系SG到固连于被加工大轮坐标系S2的各坐标变换矩阵
Mc2G=[1 0 0 Sr2*cos(q2);0 1 0 Sr2*sin(q2);0 0 1 0;0 0 0 1];                          %……SG-->Sc2
Mm2c2=[cos(phic2) -sin(phic2) 0 0;sin(phic2) cos(phic2) 0 0;0 0 1 0;0 0 0 1];        %……Sc2-->Sm2
Ma2m2=[1 0 0 0;0 1 0 Em2;0 0 1 -XB2;0 0 0 1];                                        %……Sm2-->Sa2
Mb2a2=[sin(gamaf2) 0 -cos(gamaf2) 0;0 1 0 0;cos(gamaf2) 0 sin(gamaf2) -XD2;0 0 0 1]; %……Sa2-->Sb2
M2b2=[cos(phi2) sin(phi2) 0 0;-sin(phi2) cos(phi2) 0 0;0 0 1 0;0 0 0 1];             %……Sb2-->S2

sG=Sr2*(1/mcG-sin(gamaf2))*cos(alphaG)*sin(thetaG-q2)/(cos(gamaf2)*sin(phic2+thetaG))+(Sr2*sin(phic2+q2)+Em2)*sin(alphaG)...
    /sin(phic2+thetaG)-Em2*tan(gamaf2)*cos(alphaG)/tan(phic2+thetaG)+rc2*sin(alphaG)-XB2*cos(alphaG);

%………………加工凸面的假想产形轮齿面方程在SG中的表达
rG=[(rc2-sG*sin(alphaG))*cos(thetaG);(rc2-sG*sin(alphaG))*sin(thetaG);-sG*cos(alphaG);1];

%………………加工凸面的假想产形轮齿面方程在S2中的表达
r2=M2b2*Mb2a2*Ma2m2*Mm2c2*Mc2G*rG;

%………………齿面点（凸面）在三维直角坐标系中各坐标分量的表达式
x2=r2(1);
y2=r2(2);
z2=r2(3);

%………………小轮齿面（凹面）点坐标（x1,y1,z1)在S1中的描述………………%
rc1=dc1/2;        %……加工凹面时的刀尖半径

syms thetaP phic1 %……定义参数为计算符号

mcP=1/(mcP1*(1-2*C*phic1-3*D*phic1^2));  %……切削滚比
phi1=mcP1*(phic1-C*phic1^2-D*phic1^3);   %……加工时小轮的转角

sP=Sr1*(mcP-sin(gamaf1))*cos(alphaP)*sin(thetaP-q1)/(cos(gamaf1)*sin(phic1+thetaP))-(Sr1*sin(phic1+q1)+Em1)*sin(alphaP)/sin(phic1...
    +thetaP)+Em1*tan(gamaf1)*cos(alphaP)/tan(phic1+thetaP)-rc1*sin(alphaP)-XB1*cos(alphaP);

%………………由刀盘坐标系SP到固连于被加工大轮坐标系S1的各坐标变换矩阵
Mc1P=[1 0 0 Sr1*cos(q1);0 1 0 Sr1*sin(q1);0 0 1 0;0 0 0 1];                          %……SP-->Sc1
Mm1c1=[cos(phic1) -sin(phic1) 0 0;sin(phic1) cos(phic1) 0 0;0 0 1 0;0 0 0 1];        %……Sc1-->Sm1
Ma1m1=[1 0 0 0;0 1 0 Em1;0 0 1 -XB1;0 0 0 1];                                        %……Sm1-->Sa1
Mb1a1=[sin(gamaf1) 0 -cos(gamaf1) 0;0 1 0 0;cos(gamaf1) 0 sin(gamaf1) -XD1;0 0 0 1]; %……Sa1-->Sb1
M1b1=[cos(phi1) sin(phi1) 0 0;-sin(phi1) cos(phi1) 0 0;0 0 1 0;0 0 0 1];             %……Sb1-->S1

%………………加工凹面的假想产形轮齿面方程在SP中的表达
rP=[(rc1+sP*sin(alphaP))*cos(thetaP);(rc1+sP*sin(alphaP))*sin(thetaP);-sP*cos(alphaP);1];

%………………加工凹面的假想产形轮齿面方程在S1中的表达
r1=M1b1*Mb1a1*Ma1m1*Mm1c1*Mc1P*rP;

%………………齿面点(凹面)在三维直角坐标系中各坐标分量的表达式
x1=r1(1);
y1=r1(2);
z1=r1(3);

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>确定齿轮副初始安装调整参数<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%

in1=GearMachiningParameter_input(11); %……initial value 1
in2=GearMachiningParameter_input(12); %……initial value 2(计算大轮齿面点时需要给定的thetaG，phic2的初值）

in3=PinionMachiningParameter_input(24);%……计算小轮凹面点时需要给定的thetaPa初值initial value 3
in4=PinionMachiningParameter_input(25);%……计算小轮凹面点时需要给定的phic1a初值initial value 4

%………………求解啮合迹计算参考点………………%

x0mpG=[in1;in2];%………………赋初值
options=optimset('Display','off');
xmpG=fsolve(@(xG) GEARvex_lineB(xG,LGmpr,RGmpr),x0mpG,options);
%………………得到参考点处的基本参数thetaG和phic2
thetaG=xmpG(1);
phic2=xmpG(2);
thetaGmp=thetaG
phic2mp=phic2

x0mpP=[in3;in4];%………………赋初值
options=optimset('Display','off');
xmpP=fsolve(@(xP) PINIONave_lineB(xP,LPmpr,RPmpr),x0mpP,options);
%………………得到参考点处的基本参数thetaP和phic1
thetaP=xmpP(1);
phic1=xmpP(2);
thetaPmp=thetaP
phic1mp=phic1

%………………确定在计算参考点处满足传动比等于齿数比的初始安装位置参数的调整值
clear x thetaG phic2 thetaP phic1;

x0mp=[0.0;0.0;0.0;0.0;0.0;0.0];%………………赋初值
options=optimset('Display','off');
xmp=fsolve(@(xmp) GEARvex_meanContactPoint_FQ_lineB(xmp,thetaGmp,phic2mp,thetaPmp,phic1mp),x0mp,options);
thetaX0=xmp(1);
thetaY0=xmp(2);
thetaZ0=xmp(3);
dxk20=xmp(4);
dyk20=xmp(5);
dzk20=xmp(6);

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>求解大轮凸面、小轮凹面啮合迹<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%

clear x thetaG phic2 thetaP phic1;

pc10=0.0;   %………………小轮转角初值
e=1;        %………………计数初值
dpc10=5e-3; %………………小轮转角增加步长

%………………计算大轮有效齿面四条边界线的参数（斜率、截距）
K1=(RGm1-RG11)/(LGm1-LG11);
B1=RGm1-K1*LGm1;%………………齿顶线

K2=(RGmnr-RG1nr)/(LGmnr-LG1nr);
B2=RGmnr-K2*LGmnr;%………………工作齿根线

K3=(RG1nr-RG11)/(LG1nr-LG11);
B3=RG1nr-K3*LG1nr;%………………小端

K4=(RGmnr-RGm1)/(LGmnr-LGm1);
B4=RGmnr-K4*LGmnr;%………………大端
%………………计算小轮有效齿面四条边界线的参数（斜率、截距）
K5=(RPm1-RP11)/(LPm1-LP11);
B5=RPm1-K5*LPm1;%………………齿顶线

K6=(RPmnr-RP1nr)/(LPmnr-LP1nr);
B6=RPmnr-K6*LPmnr;%………………工作齿根线

K7=(RP1nr-RP11)/(LP1nr-LP11);
B7=RP1nr-K7*LP1nr;%………………小端

K8=(RPmnr-RPm1)/(LPmnr-LPm1);
B8=RPmnr-K8*LPmnr;%………………大端

LG0=LGmpr;
RG0=RGmpr;

LP0=LPmpr;
RP0=RPmpr;%……赋判断啮合点位置的初值（计算参考点）

%<<<<<<小轮转角以步长dpc10减小时对应的啮合点计算>>>>>>%

while RP0<=K5*LP0+B5 & RP0>=K7*LP0+B7 & RP0<=K8*LP0+B8 &RG0>K2*LG0+B2 & RG0>=K3*LG0+B3 & RG0<=K4*LG0+B4 %……判断啮合点是否在齿面内

    x0m=[thetaGmp;phic2mp;thetaPmp;phic1mp;0.0];%………………赋初值，pc20=0.0

    options=optimset('Display','off');
    xm=fsolve(@(xm) PINIONave_GEARvex_meshing_lineB(xm,pc10,thetaX0,thetaY0,thetaZ0,dxk20,dyk20,dzk20),x0m,options);
    thetaGm(e)=xm(1);
    phic2m(e)=xm(2);
    thetaPm(e)=xm(3);
    phic1m(e)=xm(4);

    thetaG=thetaGm(e);
    phic2=phic2m(e);

    x2m(e)=subs(x2);
    y2m(e)=subs(y2);
    z2m(e)=subs(z2);

    LGmrT(e)=z2m(e)
    RGmrT(e)=sqrt(x2m(e)^2+y2m(e)^2) %……大轮齿面啮合点坐标值

    LG0=LGmrT(e);
    RG0=RGmrT(e);

    thetaP=thetaPm(e);
    phic1=phic1m(e);

    x1m(e)=subs(x1);
    y1m(e)=subs(y1);
    z1m(e)=subs(z1);

    LPmrT(e)=z1m(e)
    RPmrT(e)=sqrt(x1m(e)^2+y1m(e)^2) %……小轮齿面啮合点坐标值

    LP0=LPmrT(e);
    RP0=RPmrT(e);

    pc10=pc10-dpc10;
    e=e+1;

end

for c=1:(e-2)
    LGmr(c)=LGmrT(c);
    RGmr(c)=RGmrT(c);
    LPmr(c)=LPmrT(c);
    RPmr(c)=RPmrT(c);
end

clear x thetaG phic2 thetaP phic1 pc10;

pc10=0.0;  %……小轮转角初值

for f=1:2*e+1
    x0m=[thetaGmp;phic2mp;thetaPmp;phic1mp;0.0];%……赋初值，pc20=0.0

    options=optimset('Display','off');
    xm=fsolve(@(xm) PINIONave_GEARvex_meshing_lineB(xm,pc10,thetaX0,thetaY0,thetaZ0,dxk20,dyk20,dzk20),x0m,options);
    pc2m(f)=xm(5);

    PC1m(f)=pc10;
    pc10=pc10-dpc10;

end

dPC2=pc2m-ZP*PC1m/ZG;

%<<<<<<小轮转角以步长dpc10增加时对应的啮合点计算>>>>>>%

clear e x thetaG phic2 thetaP phic1 pc10;

pc10=0.0;%………………小轮转角初值
e=1;     %………………计数初值

LG00=LGmpr;
RG00=RGmpr;

LP00=LPmpr;
RP00=RPmpr; %……赋判断啮合点位置的初值（计算参考点）

while RG00<K1*LG00+B1 & RG00>=K3*LG00+B3 & RG00<=K4*LG00+B4 & RP00>K6*LP00+B6 & RP00>=K7*LP00+B7 & RP00<=K8*LP00+B8
    %……判断啮合点是否在齿面内
    x0m1=[thetaGmp;phic2mp;thetaPmp;phic1mp;0.0];%………………赋初值,pc2=0.0

    options=optimset('Display','off');
    xm0=fsolve(@(x) PINIONave_GEARvex_meshing_lineB(x,pc10,thetaX0,thetaY0,thetaZ0,dxk20,dyk20,dzk20),x0m1,options);
    thetaGm0(e)=xm0(1);
    phic2m0(e)=xm0(2);
    thetaPm0(e)=xm0(3);
    phic1m0(e)=xm0(4);

    thetaG=thetaGm0(e);
    phic2=phic2m0(e);

    x2m0(e)=subs(x2);
    y2m0(e)=subs(y2);
    z2m0(e)=subs(z2);

    LGmrT0(e)=z2m0(e)
    RGmrT0(e)=sqrt(x2m0(e)^2+y2m0(e)^2) %……大轮齿面啮合点坐标值

    LG00=LGmrT0(e);
    RG00=RGmrT0(e);

    thetaP=thetaPm0(e);
    phic1=phic1m0(e);

    x1m0(e)=subs(x1);
    y1m0(e)=subs(y1);
    z1m0(e)=subs(z1);

    LPmrT0(e)=z1m0(e)
    RPmrT0(e)=sqrt(x1m0(e)^2+y1m0(e)^2) %……小轮齿面啮合点坐标值

    LP00=LPmrT0(e);
    RP00=RPmrT0(e);

    pc10=pc10+dpc10;
    e=e+1;

end

for c=1:(e-2)
    LGmr0(c)=LGmrT0(c);
    RGmr0(c)=RGmrT0(c);
    LPmr0(c)=LPmrT0(c);
    RPmr0(c)=RPmrT0(c);
end

clear f x thetaG phic2 thetaP phic1 pc10;

pc10=0.0;%……小轮转角初值

for f=1:e
    x0m0=[thetaGmp;phic2mp;thetaPmp;phic1mp;0.0];%……赋初值,pc2=0.0

    options=optimset('Display','off');
    xm0=fsolve(@(xm0) PINIONave_GEARvex_meshing_lineB(xm0,pc10,thetaX0,thetaY0,thetaZ0,dxk20,dyk20,dzk20),x0m0,options);
    pc2m0(f)=xm0(5);

    PC1m0(f)=pc10;
    pc10=pc10+dpc10;

end

dPC20=pc2m0-ZP*PC1m0/ZG;

%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>显示结果<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%
%………………齿面各投影节点、计算参考点、啮合迹点在平面坐标系中的平移和旋转，便于图形显示………………%

LGr=[LG11 LGm1 LGmn0 LG1n0 LG11];
RGr=[RG11 RGm1 RGmn0 RG1n0 RG11];

LG=cos(gamaf2)*(LGr-LG1n0)+sin(gamaf2)*(RGr-RG1n0);
RG=-sin(gamaf2)*(LGr-LG1n0)+cos(gamaf2)*(RGr-RG1n0);

LG1n=cos(gamaf2)*(LG1nr-LG1n0)+sin(gamaf2)*(RG1nr-RG1n0);
RG1n=-sin(gamaf2)*(LG1nr-LG1n0)+cos(gamaf2)*(RG1nr-RG1n0);

LGmn=cos(gamaf2)*(LGmnr-LG1n0)+sin(gamaf2)*(RGmnr-RG1n0);
RGmn=-sin(gamaf2)*(LGmnr-LG1n0)+cos(gamaf2)*(RGmnr-RG1n0);

LGmp=cos(gamaf2)*(LGmpr-LG1n0)+sin(gamaf2)*(RGmpr-RG1n0);
RGmp=-sin(gamaf2)*(LGmpr-LG1n0)+cos(gamaf2)*(RGmpr-RG1n0);

LGm=cos(gamaf2)*(LGmr-LG1n0)+sin(gamaf2)*(RGmr-RG1n0);
RGm=-sin(gamaf2)*(LGmr-LG1n0)+cos(gamaf2)*(RGmr-RG1n0);

LGm0=cos(gamaf2)*(LGmr0-LG1n0)+sin(gamaf2)*(RGmr0-RG1n0);
RGm0=-sin(gamaf2)*(LGmr0-LG1n0)+cos(gamaf2)*(RGmr0-RG1n0);

LPr=[LP11 LPm1 LPmn0 LP1n0 LP11];
RPr=[RP11 RPm1 RPmn0 RP1n0 RP11];

LP=cos(gamaf1)*(LPr-LP1n0)+sin(gamaf1)*(RPr-RP1n0);
RP=-sin(gamaf1)*(LPr-LP1n0)+cos(gamaf1)*(RPr-RP1n0);

LP1n=cos(gamaf1)*(LP1nr-LP1n0)+sin(gamaf1)*(RP1nr-RP1n0);
RP1n=-sin(gamaf1)*(LP1nr-LP1n0)+cos(gamaf1)*(RP1nr-RP1n0);

LPmn=cos(gamaf1)*(LPmnr-LP1n0)+sin(gamaf1)*(RPmnr-RP1n0);
RPmn=-sin(gamaf1)*(LPmnr-LP1n0)+cos(gamaf1)*(RPmnr-RP1n0);

LPmp=cos(gamaf1)*(LPmpr-LP1n0)+sin(gamaf1)*(RPmpr-RP1n0);
RPmp=-sin(gamaf1)*(LPmpr-LP1n0)+cos(gamaf1)*(RPmpr-RP1n0);

LPm=cos(gamaf1)*(LPmr-LP1n0)+sin(gamaf1)*(RPmr-RP1n0);
RPm=-sin(gamaf1)*(LPmr-LP1n0)+cos(gamaf1)*(RPmr-RP1n0);

LPm0=cos(gamaf1)*(LPmr0-LP1n0)+sin(gamaf1)*(RPmr0-RP1n0);
RPm0=-sin(gamaf1)*(LPmr0-LP1n0)+cos(gamaf1)*(RPmr0-RP1n0);

%………………在坐标系S1中绘制小轮凹面的计算参考点、啮合迹………………%
subplot(2,1,1)
plot(LP,RP,'k');
hold on;
grid off;
axis([-1,21,-2,6])
xlabel('X1');ylabel('Y1');
plot([LP1n LPmn],[RP1n RPmn],'k');
plot(LPmp,RPmp,'r*');
plot(LPm,RPm,'k.-');
plot(LPm0,RPm0,'k.-');

text(7,-1,'小轮凹面啮合迹');

%………………在坐标系S2中绘制大轮凸面的计算参考点、啮合迹………………%
subplot(2,1,2)
plot(LG,RG,'k');
hold on;
grid off;
axis([-1,21,-2,6])
xlabel('X2');ylabel('Y2');

plot([LG1n LGmn],[RG1n RGmn],'k');
plot(LGmp,RGmp,'r*');
plot(LGm,RGm,'k.-');
plot(LGm0,RGm0,'k.-');

text(7,-1,'大轮凸面啮合迹');

%………………传动误差曲线………………%
figure(2);

plot(pc2m,dPC2,'k');
%axis([-0.2,0.2,-9e-4,1e-4])
title('传动误差曲线');
xlabel('大轮转角/rad');
ylabel('大轮转角误差/rad');
hold on;
grid off;
plot(pc2m0,dPC20,'k');

plot(pc2m-2*pi/ZG,dPC2,'k');
plot(pc2m+2*pi/ZG,dPC2,'k');
plot(pc2m0-2*pi/ZG,dPC20,'k');
plot(pc2m0+2*pi/ZG,dPC20,'k');

toc; %………………结束计时语句，与起始的tic配合
