#!/usr/bin/env python3
"""Private two-GPU, image-only 203-video runtime benchmark."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CKPT_SHA = "712646e9e263da018239dc46e927905708d31405c0b1774d20a260d85f480987"
CKPT_BYTES = 6_375_639
PACKAGE_SHA = "3e3e96fbd5bfe82f716cf3e46e00cb83ec67da69747e687d021c0e412fbd9f08"
EMBED = "P)h>@6aWAK2mk;8ApoOvwm|AU002?v000XB003WKZDDC{Utcb8dDVSucjLy9=y&}J1UYvD%#a*OGalQF>2n-=?9Ik;d@^>jy9^Huk)VW#1Q-C+V|4Z3Z$0|~fDg?~vU_>#5ef98y1Kf$s=BKB*S~tPt*aLodGR7EZi3Byb5#~EPr@+#UAZk5*<ujn>+~`kmc{BmSZveP@Vg&>4)S7|Rar65f@N6+%~cl6%k?ImH^FzO=fV6cn_q9ryl6%zCqKc<qO8{GD)>WsdAZ7<@=ZBUo4hQ7ydIBFP6GIF&%rXSo4T5h-sPKMI9!*DEU32^>%4(qRarK{i)(7+MP6*S4ZvMqZ7+uNY_%FTRXV@Ui_76+ThX8FCR>%8byhUN->8oO8nLjon)<@N87|T$t+VFK7tQI*;mzyOra3wJxz2)Lei4{1VC+}xw7UM~m!K(wD*MMauQEbJc86#<hl+$acz2oCO?5vCg5PPaliRYo&Z-(vQ8(#o6{JusT?FZN0V8d$(guG2J)1Xm@ZFnlza0c=QIrj#Y!KA*vdSuawg~FYDsN8KX^}6px(POEb5%n}2<M`@A6DCfusBCJ^BTHo^5Q=D8fI{_+OA>NFL|<l8(^UT$94kc7TGGl$f~r-VA8imdXwg>^kT)cod%n#Tx{nwfzZPUspaHkS(WP`NtWAYTV+WCWV0!&#(}&(Inig;<tDA_On=Yo8~u8f)>qKC{``Ag7W#b+gVXP2&Hdqkp}2PTqf-RP2H>1$b#0#9o3Bl_-YoN#fuPCP_B+r~b^)}+T{QO_M3cPys<<Bnzi$9HfQ~`%$8>{l2f<sQ09+5+2+%LJe{%BmAAbBPdHe5Q1(P8B^z8igtJm4*+4<|2i}dvDv-6jqFXs5<{L{}*Kl$|Zv&GA^SErxMPcP0s{q)u1{4_m(eY*JUljW<=PCx(b(~w*H-GBY`<}I}MI(hl&C&}xVpPzj5r@#IH52CX{@CyEmPriQpm*m^uzxn<fEEf^=hxsBL1fkG2eyz&e1csO4_X4Iq;S%)j9sYBV|J;QGu3wkiYHkX_BxVi$IEhc*{O6BvzW(XWH_2Due-9l3_1i}P>re3P72smNO6xlK7S{Nisw%7K&)WhqPe0-@_Y#JGPV+ijz#0Q;%YcA@0_${{@e)A-Tmt^_IWUs6%2Qa%H+i`tA{!Aco-DFukknV{`6sU<q>(X^MjQ;k3}7vCPmBB#*UUs|ZY1SndV8BUS43W;a+4KNSY3p1kk(MRSgf+K@RNX6#Kj61PYwV9V9_dFUo6tGlpG<{(b?(wtKeCHf8s%K5r(mC!NM}yZeVt@h}vT?stlG>q3^G<JN^bh${=a^CJX3fl^0p0Uy^K7&SBWgRhc%xGa6t;V;no-;D@r1A>>P<{Sp2X#le@8pw)r_v%*<%=585QJbR`FP7r{z2$%r}57g^pXkbm7a-Gi;T-46241$|<wFT&8F--x=0WP%JpbZQUC-}cK0GK)a(;5&>8E_Zi`lI#rBCjI;QBQuVwy<KeJK)Aic}+iKsd5j^E^whBXrsk;y{RJt3NQz&Il0d6wSJvJdoYlyo<!jQSHd`iDFr4Di>XM9$+zjM2B4k?;opid9syH;l@*0)J<s!yJ6yu9TBK_nfifb1wZ`w!rplK2-DDY#9%v|7?r{YD)Y~P#2uIC&13zJ86KAa4Fl;k+!ia*)dbC^+1!Je-?G7r&kI?IF1ymdfPU5~2%hk5Niu_mDPkmp^BP|1CD#|FfrGV1>*~ASA_6BF}^s}F+2f;FQa1$)^B3<S0X!(b>KY*ypHY?!e5tbVeWxN+S9#%gpjQGj!=GmsP3#;1|wX-dt_3LPz*EOsWpMc#DEz=XH;WzXL`n3&r1_Zd0_7Pm$sKN%5`9s-!ixOcEj2mF;KAVsQ54`arUo5gBAtuO69#%IjVCWIo{`dKHhU<N%tXbB!-3yri-Co@X&#?Y8ohD@eC||u$E<{q?0QZMqe!(Vw`Gxla8gO=pjGl-TTOm+lLm%Pa{)P+r1uWM&?2XHEwTP58R3L**f+>C>UQb_u)nk(*62Bs>i1dW7X1u_ig4D#=r8|5Ad|^@J3XQ_iWwu;G3s8r@zX-JhAJ4KEgeVe*<_)q9fRGxk1R&%WXYurO=5`xeQH(mEiqI+jh2IY$$avWbF+pEy{B~L4q$f?aZLa8d6F&As<+Seg%Hryd9_b~vOJAtRb=Fk*+<vQfbtnzHojDC7BB){fQagfAVqSR<pc@6vnHzl<?*M_CayM$EJl*a@xF_OINVJcL%oq8*;Y|*7jTyHLB^%ri9zxpuvqd-#P9K?iu`#X=8cosA5*nZoU5|riyIEyZl1fLT(X6!@3R<XI@GB5NaVH3IoV0|X1^u4a1T9Fg5k3ae83Ymab8+8f^}t=QaW6K^BT|qi)cW*nhW*1<K&^s&5u61;SsCL=Yb;mkWvx}tJ5|nmsvu5L#+Yz-z!XR%fYC(GQ1IP=r5@ClK#C(eq$;@Fo|~WkM&&d?u+dC)My(L&H|(5;@^6T)KP7+t^PhhB{=b6BBsh&-$U(?}YAx<4XeUmhEq2GvFKd^Z8}gYN+L;B<C&8HyE{ShHW)<xqWuZcx(PUjce^ajVIjH+tk%Kt*z&$<^SmWTqvQdSZWITf7z8<BE1ySo&Szbr<%@bt_R9zyf94AC&j!-E30MRqOGvX?i>EEk&VYajJ_hGO{ycO6?kEYomQk2Ai4t)I>x)V&}q+OeqK97wHtje;+)q!J-KwE~LHA&(Ta+?JA-Y7&12y{I{QB+#R^?>z8c}+E<c--!d-*>u$bq@oO?r3tV%Nv|AC|Z}Rn=CRtP*K|hEf~>a@u_@`s94;RGPT@)=bLYBArv&zYJQd9WWDTD-m!+%H}3YFgu|hk6qpaOx`0l&ZR&i%=9dj9?m)*29uJvwi~@oZts%J`#AcZ@Y9QmTSZyGfkac2&jn+{gB@0gxAu9UQTN=e6_(OIt->8BAc>AXxzR6Hakf(~x-=zydD4`PC*{zL*s0*+1g|Wl=#wQRj$e00b=zO=hzLp#i21pt)mMc*7L4m27d<iW>Q@E4hbS2b@8G5(}Ja^t`?+JPuK*JJRy{wR20bZ&_O-$AWMxNXC(HRZB9|S9?6ze7@PmclfDn&RA7QNnp12lP)15W~Nkcedl0K!*UOby6EE6@+N#Wg~HYDB`Syv6;d14a-0<*iJ_elBGJ=0+XqxV%7AI(&(Q6=;>E9z<z*%h+Tw2DDi7M2$pF?W=SHd#M?ft&A_c%3Q{&D;UEi18NcV_Bei!J|2Au4qq}hP>gF}zy+C`vbtdjt%WIRMRs|3Z#qMJjGd|@(Z~~S+o%IwSkUwWgsprI+t(tD_e`Fk%8(w)GW3jA&MPuu757oA)(s08`1U6F0_3D6eM7OC>K?saQE?Y?Z0JFU^tcyvo;nMOOgDgYboW#R*bt~?#{psH2qlL|i`W9*ass)AztL;zrOO@o@lNKzuN_9m54i3P08rM<<Z$9-v3ANa=TcI~(k`6Gu$c7n627C@FT+@r1nD|$=2tDABRi(nmHljygAR0)RmtDWi<*tRct#>-HL1uo`mAeI0EJ)HC|K8k9Sk(;ad1(VD=CrQC9=wZ<Ke*xZ<m7;+AxvKr~a;~I!(3$4wRu`jS3)Fi+x7Oz)cN5C)uAB2-stcY)UE%RGJE_n$c_G2{YMaH`<kFir&#0$<>~L+9egi0ZBl#c4Zti%i7aE!{`I{`Ka<q%|}8YRzGOvM;UsC(hE@%kmDtE{owWYXh~S-#qwi|T>Uy7dvk?2I}`M*5E1UXegV2GZ$YJ@TS-rqP)+a`m_o6qxuSH(ZJO1Z(jF4?xR<Puka$3129sj415d5-MF^}56Z7{;z-Ng*O})U~37`B%K>N+?ar7>&d>aEX6Kn;gVze)}IJ1B@g5{PKi%6TKPN2bYyN8xJa9#rYE4EhSLv(0k56_;FWDufw8Pc+i2o~6WL*<<C{BWm_X6_V~`waapVhxDv;1{XPbSppNM?QxH)kvO9$Dhp{TE?<CD#K&#0+ZeH0bWn-fMztO@C@e^Yd)H))XpA~Qx@3;o%SY_mRDJm_3rM%bg>IzxUjWW3+A>=rF!f)?QqxGAdo}eG4H0zD*Fg~X6&8cRav&kfmd9lE8MIA>h-pvvsZ>6`vr~IT}U_OUACg$ti==1tY^&L=FJHD3!ZHVhP2pvTQ|W4C{Se$yYWpXNT4C>dhCTASKgGTC?T$%7MEEh&BilPdsC9)LG=utDP;8KHp?%sgpZXKe2EOK5p5bU;J7{i@xT<i&h96C;)aWP97Mx4or+ZDBI4Ce^688PRF|lvk=QT|=|CZJkp>dY1c)8P0dH^vxBI+UWOo1njde=DW&?g2dvY}3bBayQbXnsNO#!Y>?@`*ZqoFmSqhX*|Da!HFzSTngi1dgWlrunjGILiWHsmiUBe#OALvPJ%`n6}pp?e7RfP7(90#SKu3p77pxlpqZfvI*Oat}!AdAiD?v}te=bygTHDYk#gDX06VPjz2!S4~chqOlR1ord-qNFT1nJiW_^Vv)zDcR3VA`wUAcuC)|isVGTe1wM#JP%Ga=?s)?HPTUdOaUhw7*iksM9fQtO)KGYh-Ma;K!z4JTAmx^bWI*aqT7&&8Lfe)Ow#e?9C~vYgP!K*6BFV*G?sSQxLmM~=x#S!Wlk~*ZESb{rC_+dx5E0HJ1{4Q0BDyE+#4~^Fk>^{`GLl%^$nUWM`Tn*kH{^7|9xy)vSn|elvSVx{#6d;1-`2!c|MbT<!ymr;_S@jyi~ARMfef=_Em4JymSbKo^8!yv2??yS;kV!m(Dqqxqn|KXM$^9a!a(svwRtd!pxl?gTDVzVE&tf2u=Ig~pSo6h?OW-sT^+3Dpve|EKc2+^-I?ATI=i?}_-K9-JTNH$V~pjzUeSbzfyV=j9q4O>)K*9Fqr(7im14PIx1M&Vm1%}xQ-)H(FgP8Zo}R@x8E4t>GfT2pz}G*t<<CF=yw!4-az84y!kSSbiwS{;_kutP>|FF4_<V9*q5dT%6#osPhF%elV0=*YAZXe-@A!_g@6+lEC01<Lx~zbToRHGXb`eL8!V@vIGPNEP4a1kQyK`B)3jX4Hv07N5u+p9}sOUsX(LICwZ;<@Onz7EB6uIxtG1^_4zRk0IMZMSgJLvocg3Z9iybSQU3=@HQ2YZEQU?X!D>?wT6Zyabgpp+*dc@)_Sgd^U8lI_~K`H&<?4-DpbojdCvKq32%d-`aYxuZfcM?w~+jXE~}MT08|ltc75qa$J$9@M1leh|m)T|?y_ZU;qG`tpDDq>q0-56(Os^F|{KH3)u<o_Rnz`DIa7*^~<n>Ge#-brvJYv4L9JRM|~lZVw3VY+Bk@6&<!7EJ=gxU|h^E=_1GOcz)<nf#T&wStavyL#hEkZmz1Vz5+?XD~vk(95#U*{laP~?o1{L<o>+QX70>dN8Vs_bzg%Fwi*IjK(V+U7Fk-M2OQ!D`3oN9lv}jTgD8mykHd@m;SWgLRetBG7CbE8?vS52L!y-i&wR)aDIsbl<QW?xM8O5Z-Lm581S%KPS!y4LO#umPcZW%-?g=m!;9Xb;#5A<c*i%rY<rvfz+r*dc3E9m(cqE_+=hZMns$_>}<P3!Ca=StoN7_EJh0!@dq$8cqXhRB}r)m^rv_qzX-=@ZSb0%XDPfa@gqGfG`OGjFxx+s;+aFnhDmy5`j4~*27727p=1V?T8&93ex&B}tfh75$>F;fwb(f>)HWKtcV0KF)?XN%o%(F^Du7?J~Gh#tbXXb#w}K`Ju9Sq=DQ4-kWZUC)K*3p@iKH$L+hV13>Js;sAzd!ijN0mmTbiFYI=)bo6EKf*KoJYA`4^ZXCr{8VMxi7Bcaoz4RQOSD5ssmLR-ctmT&s_uQL1qz;+r8EfaegT_7mC$ORFk>;MlgXJ?^u1#XXH`$yXA=f&GEDG>;P3sW#LTN`2iadu-rGvvba<45pQ%u^_7>C3L}C!2ruCJW36w0*FMC1uiX`tL1`F`ACC0Tn<U@ShDR5zAZL&0DfAnHDH61H3T`JfCl3L<P9R5AgRLfcL*8v8DIMD+vN5TU<b*Y25^ZZAQllbY1Vh|d1Rw+pF2r^jbm$1FwJB^WFhG79U4OUofMk?OAq?nAfaOedYavcE0$(kEk9cDYfrXvnAaP<jGe5OajC@!yuM{j)-j-3T=nUlX(d}_te6{AJdY{DeM2EtVLLblr2r$lPWCkBQdT9RRm>Z}17DBZ3=@`OGHzzzHwve9vQqz!|lt_vy`Qn`(($rLJ=OS0p4EqM5eqQyH{?dVu=CshBA%m70W9Z%6ukssdoJ-qWDI(9$&I6ET4Fb<-6yB6F%3#^i7FvsO!De9LV{zBfH%Df2OEl$>+!2;tEo;-)zs6Q*_$imW^<>W4e7OQq#b$8|rWs-I)w1LK*h~Y+y>~m^NexK3rd-MC8e%}GgqPF8-8|EGK;eJ$YGZuf~-+SmzKNnS&UU%NUM`PWFJSMIqn06}jyw?Ay8X_@&O^dac<(6P*=hjFv&p@|vjbKJZh}A^Af97<IUa6UQ-MJb<R~)cXfrtjap2uv^7eNLFEA=5!#Y3OJGx!E{L()5do3-ei_M@!MG3L<O9}QsBv6<>w%Nf#{8e&S_nexh~^9K3|#Ma(F#VG#`y|I(XXXe$H`#OwP2ZAb-C{MUsNzdkCmu4ArN|upD2v=p1LN;`zLRTG#TXQf%79GVqu&j-mNG}=HwR;*uC`R9Fu>hQokT;J`0Y7G#hUc{r!4<!pIQnj9%$A(ZxQA4)2S7#u&&!U1JH3-)XIhW<=yU!Y^y_w}p3&-B%A7M02kqM$HWD6yc#sK`w;p6vga@^&NVd3aE%#juJSou2B)`G)6hdO+V^NnhDM&r2z*(b$Sil87Fx<nuQJDP=vVEjYu$#H{p>QV7?TdVa_<O5>bV&uI30IG&zG$XF`S;ZQfv4y1vjoa~qg4Vn`fQyybdXh$=2X`?o+DsC?8xI=2C4r+oAT74cd^Y^iv-<8R#8({7~=L-aZd&w_-jl!-(*XSue0IB6>JA721)*up`e|ve*OVw;kohtOc&_}g_0&KYO>{BW$V<MaEbyr@3tTzoH!r#J;KNOZDiluC<K_vn}nRqVUJJdAXyjLO3YH|h3h}^hvDD^T4!q+s-M2J2(P||PF^nD4*a$U!AlJ0THpaAD2TyDp4Jj|H)yx@JM^Xcp{&->pxXvaKD_;N_WjS$4+R17mepxMfbskP*F&QlJ;l%JoPPymSl|>zQ2URdRIS`Sm*^`&y)a2oL|US4xjj4GV7Gb;x=|F5Ox2Efx*TCRKOUqMqwmxEtilmRW-Q&}Y10&3{Ev8=d6lIL3%bSQD?lD1C|QDrrpl!^9N=bXagov7^{f^BP(vt<WSiQK8-TbQk)3@ye(B2qY74O?UXEuiF196c!kYS}jz40S@39%&-=LwWAC>_(7>@wm>n$jWr$k9WLFchw6%-jN+G(P{V>3f?^wBK@JAjtMsnJMR|4#uMS;DRmO3%IB(cku1cfjk8^_+(rA4Jo$UNj3QxdWa@5cI7N<cTl8Ic{;h-)dMP`6F3HFJU~dfr8J@KkHcmmnaPZ!6!Dy1cweGk#~!8%y7k=ww9$sL-;<go4ys<$dHe#Vyx)xAR)($Vb+Xi$>0I^sDV9j>jv*^8rGO?Xre2p^$kedf>Hfm{9O2Kmh}RRu;QbmE^pk^lY<8>)kQ6yxd}egyC2~PFz<6*rQM?2O`W1Y*-=xr$eu9Co-W9021*n%9UytLg=BcPYibN=4kmw=XKY_jFu0?vCNMUXM<yX?av0E@Rw0JB=a;Y_msNpQS!M@2X3ig14-qiXqO8FBb${bX(9G6;6!#zy_Yp>$Y|>H-Ll<#5p6%T|B<H|UJBVR(vUi-%j%9_wyM>`Wb6Yqcw`_6sb-t<l?L4v`IVr>-(1j8E3zJTeyK8XJyP8_3%UzPuh1IVQc{>d*kG`G8Ey(kJppe2C_3kKrSUdwPcK6C%C3g-|tl~Y`qlLA@<b@Na#N9J8oH<={gFxmEE1BbdZ&eQZ*3JNSxf`J93}_a+i+7n<+NjiKccC0c-l9fypuj%tll@a~m4i`^R>BRYU&&0)jQm|T@tZrAsT`mo)=uVW6PdoCn;1`%)|0b%k3@vrq7NUgL)EcF=L#EdH`yGd6J*){L_vdwP-TnlJo8u{^106j>L}rr&h9p7VscA=W*B3qH0%KRMa41bw!&)=Y7Q~dWtL*br+Un}k+wySG@GvEce=ext|)GJC#y{tf8W;Z0<4X}w%`mRa>(s6TWuo;9g$XPn|xq2#r(?XwnQ@qxaoKhjB%$?@FE2wHpJ9NCerBh7SbYq?h(dUtb+g3sRGZTFE|}eOYrR#I+|?WB%Q63$fBPz6hmCz(cAsnFVB5Vn04Ck-hfWaU!Ag_&{<BmtJ<t3PagUP-M}Bv`*!RuH$G3HJUZl6D&9WxRhCl7AUUxl^Qyc>iyUUPC~uRxsFFptX|BTfaW7eORL;>VrehCyHwb|4K`50rm~r^KgOd5gpF<AjsCRB^h&*Y&m40X2Y_^S0?ClA+D`nu%th0pn(LVRFgGau00CUY>x(bZNi(;_Z;ANE|O$4<7d3^5VsdBPg&_BIlW}3DQM{YcnoD_uUy&9oMc7pp7?sQHPHMEaP-Y;7vHih2)`&WDHf@*DZatfNr(RE%^AYa&%nMsp<@MIxIoX%mVOgDfg&X70>=bJ4MuV~^4#$7q*?(G~uhrLRrA<pyQ08SaOmTA61>$UGoYU1oC!Gmcy)X<qlLL`6-%s#pW$*R1ihuJFKkRcaj8}#SGM|atLOHtl#*pjt&!Ah;`1Y3hOQE>JaA{qQr!a&zBkrk(n0nLTK@reNDt6^Z|fkqZOt;-vV^h|OF38!SMr5=@v+JU_|Z%*7OTI1;%5sXboQ1J{)d^ESrQS$~v`(ovL$V%z~%3iQ7w@qJ)MV7NxiISKDh(R>Tk>)WYc2}&dZP$!iAIb3MS&>$GDFZY+Y}ga4?SR&aZa->qanK@~!A(Db3qO5_QFra?W_)r;k8g-GWC=wXaeRyOlEHi;fZV@lbZ<`-_e@;S>E6Rk$Ge8>G^L2Y%q1iK0~!$eyqMGjSOB*Z%waT|Zx^tO+7vI7fTZl*MSZ6m$36v^yGm<}3W&ND(-Dkty<NdN;yifG^5wakGii5~k*(lCr#`<}QD}PLaUX+pVc+4{aBOF(W9UuDo{~zoQ+_yetm_?K#x*3VPWz=-lP8X69P3dWq(eWQq7>c*1<#b#$iAr3hdl+7Vas>&Taxe2HVG2!dc)D$zL>&(P80sSi*`UTD+|^(QE20*xGbth5@6v}X^ak@<8dN2ZY$k3uuFj?#`P$b=!9zkjP#2O(josDhzP(cE?|S10L<JZWsl)PCZ#YfTNE0lKTAeN%Hf4gLu)N;5mJnyhL$y-`%!5$+#zT3%AJ$``&u~`B2&<1JnOi8jN}^aJu74uAdm_RfOff=BCBgBYpHIxF2!?A<M5xQ7%71}m`530N`7!aKqrWz?NRR2$Ev){NzCS^rl+6|jkZM%O=Rz~=rrb8(>!>95)uG<jt570hQ49LD7QM0$zZ_5YysN;nbi-(nLl{(LL4Y4(}h%sJ1k%XbP8L38AS^BL|}@8e+y)?>ilwD<_bjYc^Skps(k0;nG<0M%zzE@I~N_BQTkThQ*hpwQi#?VV-(Y`f||GFh!~QUu>_rh;a=u(2Ez;0w8T&i0>^y9lGb}wZS#L;DZoHsn<1zM(uGXaI={=F(4F1aR@oBTs!$+B-5;A)svB^t9ucxb8mMopCJks+FB?+i2f-<N=TJlD0j$*vO@p=E$Ra{0J8Nn{fJ{%Zmd>m*l`_Z!O>pP?+prOv31PDr!P?Kjc4-y_DLS4M?oC*-+`zs&_Gi)Afh^brM9ok@g*IUB*2qZ5H4gX;HKfm-3uFMGU}3~u6vsVvK_TB$8Q%8RR)t~Qw?tsNWeK?=J7Z`XVn$L*gC_`f^n91T(LUCh6MU%cv<5$;h?mjnC#Jo=!a7<lEyY`jT(FMvBG}|AkR%gyUy?j74o9Nj#m4#Ja(|}pii{&Bm<U50L9=n5^z$q3YG>q;c)RvT9kq0EH|S{ZLSf){u942ZH1l}C5;kCzBhfwtTQ#}W@ETD$z<K0o+>~C_>|g#)G3DaZF)v>G>&*LgJ_DWt1eDINFtB^Gg@23fdUNj>+Ds4$S36Xk9QSZ2o6#z}BOkFy{1mA}@g_IG*k1@JJH^$wwb|L4wO|?8u}f|0bf(YcCtDL>A8;Z?Rw)n>pRcl)9ZP-HC9Zcyov{WD&tkKNeF%7XpgUAmdLO~pZBeB+APIAXxa~!y25FfiJ7bk3p&~T)`c)6E+03JmfRORh7P8CRIA}Xjq5GWb4cZ*&*=w~|+?d*)W(E^82YRadQjU#jr5{RiLw@LN5@NSWIXdzucGsbhHA5adVvT4*2s(k=)`Onc<=Oa^G5MJ#+NSrWWJ|4=hnVuyff(HzO7C^003Aw#2g0GmMz2e<f|0$N0d-Oq5YkO59j{5_IXYI}N%tU(nX~$iT;V%-{w{bH;H3+6+8koeoqK(P3pKohCwKbfeAa4#(;kmacpLn}3_fm1+1QMJfXNn_>tPl7YFm7CM#!++?62x@ha(^8Yu$Dz)4Oas^u1C^mlI8hs4KL(H`NVA3@xl*nF-T&qwlyO11xvR`R1X1dq!w7xLxPgig92P{@Lp*#yo%SH5TKb6&cx@fyR_*!r6f5lz+bqbw>I3eW>%vzwbJ2*?Td@0fwqLcVp0lzM0~IbtZ}*`x8VSr28gFT!GC5b>o5ePEXehm>c}(Ocb{5Pn3%xGdG0CclzaCzufheck!=-@#a869Nv2#Fl#oaD#heDy%$yB228xdOOf~FLktoS(1>4b+{c)NjE^?Rz6}f-pU}-DZ1Kia-JBqZGJ|2ZBR+4hg#Xe01@)-(G(45u9h@uj(9TGeWkY5GPqTm?Q%^E2sC>p<8e^-M)#%71WZZ-rRynmKw(}dc@nOod;DL&1-Z4$RXY;b6r0}feA3`Tb^xjGJcQB`;M~+J;afHpPr6~aMpqA(ab9mtobv(T(SKD>AYb;@_iwAE05y5<D!RaJamATjJo9x)TIXE=)guz3n)~{F!qzRL}mq*W-q|rDuW9>e5@_1$8aRyD-DW{1vOU|T!Cl^3;^w)Ck!Zah3N?!!$d-Eyl4QD2eV~Y9)ln9#JQX0|Kh1qfdJhJ7}5*yJYRcRyv3L?Iht4E9rnqS1z?TZa=Ivd>oxib+*YCZUZ2i;6rF`sm!Et*mVV#n|U|2WEZ8`Dst{Yd*2TDi<hypRD|G2MjY*dUO%bKYyQoP)Sa4<qIP^qqxbN{HgC?2-!{3Jw^N1A`A^97i~~G1&oCAJ(jEwA*3Q8jUE!@%9=%Ilqx3)YHFHQ9;IIBh|^*GyC;?wo{;2Un6)vUxADsVK>8oGVZ(xD8G*U90VFLWeAe}i;6|Q#zpp0?a-+5_~J2}pZ8_hq(Gm5I1sZLx-^SJt|^$6^K2k;dooHXzVR|R*BW84ZM`N3uF=6aICN|W9TP)<hLKO6WK)n_cjBjo3Zw=?t@2BXz#-&gF|`#_70(E~%Y>k9SQa5=oG2)h9Kh8Fh1f+W=0tfjh09f@`!s#+6*PIQIo02;x2x?|-?G9EDnGC}i)eBK^M(>jN;#t@BWdC$Q{d6m{JN`(oA7?e-z|>)M#P5jJ_d_e)Ig#uU&~cpRdrHlul1F-19}>Zj5sOpVi6nh$33PJdF4xu_CrD?7Q5qOZS{<>F;!JJ-r0<=y-jJA>p!!8CGNddH`$YzG0I9kMo-i2Rx}5@guB<m5eM=$w(b@$H!K2YVQ!<baYd<Wqs{p(MIPvnb1NJPAu<w0%n^g}6&a1r!N6TI`C<d?p<wVXNjE%o)QWx70NoP9Ux5>!X%<W_N{{w&U&C3%po^vm5&#h`o(M?ls>gpRLl_TYLLQfl9KEBc+7`@;cPJo&gkqUr?tIM#5zJZaWeDf{v&JpGM3{5$br_WBoXqcg&aYaxt4-AQPcTkH|Fvc`J>piiw%Ew$0Nah`tHnD9k-dn+AS}L?pCGDEq|(t9n9u>;FF+YEvnnCEj}+83IR;_SC;lbvH>A|d6*<wc_=4^1J;oEkCTU4FC<^*&<v^xSymE}#s;BSogBp?pXs9l7rVa{b{h))a!XuOIUsy$Ag5YErQxsBppN{M^0bJ&kE)yeHBvDb-M8J0)DM6sVDv90s#$QR(ZmKz%dsQj8q|1FL4rPO|h23A)A*b^ulxQ04B;nsqdhdCC6FPbUH8`R>BI|vE7AiN(EoEhF$^cVJ6MdK_#>D36tLiWk-#{0~n4aj~AHLZ!;+q|<eK?_YhCBb_(*7_(#O4VJ0cl>j>DKg?1iAnTI~2~87)|QbJt#7PU^YSCe$%v=O|%&{R2@FkW_J%ogxsup8{+nMi?lomLv#|Em*gW5Zky%sv;GS!*%*c6tZzo>ApBQ)AlFX<x#rnwfor~=*sE~kR&_RMp|RLN=&ll8@|1<Se~Ea*w$8*JQNzB|ilsnq#H*<l(d8_$p5Tu6S^w^i%f0a>Uag$-ybZBO+JPB#KdSS%>e;|y7NSyFzJO}!%zZ$*X-{a5&H%x>X98THJB_$2pmQDCzW}4KfjkkRCdZ0hmK{~B)62$$?0Q%?;xP*rLXJjXAQNGGM{W|MDusQ5?(T`g&2rB?IjHE)LZ2*aFww{Ty^(fdYV*41lN~aqrhe$OfP4NEr%uv5Sng~DQYQzg6aK2>vui=}a<^xNm_H{|>7A3KJ}wQo2{3!8fH#}HWbouJ{Uj@xorNy))972;<|9hkxtqdB4%<Uw?woZRId*??KW2;TIVT&iP~*%amp<HzdzfQd5#rg^#(VBZ?PeUA;`zkOeezpZ#E^HKH|4h-dE@z{P2`=9`UMg+ko3IiteO2`+D-yb_w%t>`Hp*<eOnHGTo*ThOO&W-NY8N<n<wm_?#K12D;(#$ps*B{k1a380<=8cg+%4s8>Z%GHpej?o`T7%<S06y?MqXNANI>3=N%{8w9h1vpJUq7*ruFrWf;f`xg6EIy%axlz9YWp&5{Er_}XL!?$lR2D9+s??ITG^9Ax!CUGLn@>!#Ec!X1bh=;tnE{lZdDh&O&vU}UVwLB@_eR8JStJcM`UG-~X<Bs50JN08D8!b6f8em-D{@1I#eCplxQN2=H*wd>>)9RQ6y5BicYe8Bwq@O`kb{r~w~9wT`hif7bKhiBb2N8tcTv0=kFTg0d4o1~csb~(^Sa-7*&+)49{o24eD2W}AeEEbK$kZ6diTqo=as6fPy=lyxNKHmSfD>S2@)coj=O<pAHY+Y9OJtX$w9Mc??_Dlur-SZspn|3!z^Dyk?g_7u<i%dDxhrD^&UzfKarwzW3mRz5AAfw{TM|+(9jh8xMYDzKTx;v|f>V7q8|9TQ@A+0EGho4N1d-D$FYQ{I(ZcsRi9t<Mn02Ymx=p6#Txy{G^mK<}0i?j(L?d)|(XeFErWa>S=I~{A+9L6yKcD4oQ9N|KjWLK?wKYIr|4b%0;`fax%>aYDi6x@ci2QVV*C-Q4HMAo~5A&AT@iSZ~~qwmYCBjnUFQ}1Ea9`yg>MUNbk6B7GuQn0ftjc|HVm#eLYt<t8-t@M_raFIm4$$2#;CPGZQnsNi|FWr9Dvn${SH<kNtX3drbv_maR2mVQLx@YlV`zHG(O1REQcCeGK0@CVR1@659q=BH5;Bu8yZ9$*8BayCeOX+Ybt8$fqb}ZuV*c2gyAQJf0Q5f%1a7L|<@c{O8^vMn+Sf#?7?SAU;#WkILqDPwFokMuK*-k%w<~u*MLwEs7ZA!TPMmyPh-p};H2-t7i+ie!yJ-&}%?9<OjH~N`jf3WdSKks1#)X)EJW1xQi2aJM!U4MviP(L5VNT{FpF&66Q;}{Khwe~*7!#0XO(1@s?f0;2+KOftusGlQ@i~9K|8yWRu<Hu_wFRG&G6+UL$KeW{YlK3Jm-0~^2#tJ=L`}X;#ZH!Oa3}qNS#=Z~S5kJx{_<=j$`|oxJ6Fxj2w#z+nXZzo~oBfM-u#d58eeh28)ZOXBccc&9g+9X0^Do(LK6HmMjlJhS!Ol#KrIx44k@L;YFQ80q>nk_PxE?L*`(hsH`y8-Tx`DbMFhw6U*q)uR#F%2rfgc%1p@FH<iqc^-^RFf@x7_En<+{gX*e>uT2~9C7ug0j1g9oP%4zzK4d9)d`cDe1Ds1!0=`tUw2kY>4Et>mt>9nkhi)L0n%X#30k3~gWLD|7J)Tb22$&fu1l&$&0Ch&A^VQz5iBtuKz8HtSZ)!c;n^3L^IOUdQVM(U?tj9JDrBOjuXnsD~}a%P=bomDob^l8y}Rbpml1^8HaYb{FitBnT{BU>*+p%S~d{ABEHMw`>=1qEyLizLp49W`+Zf#Keb|fg(JHtLf_&7j@zCC)Dw4mnUxWt1bcJ5c(LaB<?Kk3MfIjY22hB#HkXS8-yRA2QPaiA4nSF_%62+aPgIsxGEEjp=1|n-^m=_0SEqjT^5wUtu}>6bS#F+s-lkfqC`y6GQ--m>gku9emKsn6Y7*XgMF-aZ_Xr09eO1t!=RMH?1;r>w-i6uQail9(Qn#0B>*PqoI312MR<9Rp3!K$)){jSCy)m|vbS`)UF4L&CZwEW)aog}@)Bgem+$y7yHUuq-0~Us7`f+(Y*Od7j+TBMe2YnBt^iWhUGQQ`iPf!xi)_jFPG&q_41dZnS35kO)ALqSn{t&yeY%Y8Ds%G*@{N`ma9H6|WV{@C^}WbpXssy!k-{vzIz8(vhMwR1N-aej-0i;VG)5ZSp4N1_6UqR&S`lMQ=xO9({!pL(846xn&DW2)*<^{*wGoR$(gk{5YkVp7Amy;D4|a&ls3juvYO+Ul>;^Pez{Lq@tRo0z2pBle-<-3%uC3o3oN6^*r-UhU7}!cLXzwP4O50V_ay?;BY|ARd7d!R*uhV*-=j?yYMZ1p7N|$-ToDKe|dnsng3>p84C^g%GC4J;oY|U*syehYq;>fTLTq)!QAj?t2Cl#D*BGx`vXhay2mm7U!(it$w1BY)LZ?9c#O@;5!#2MKSFV^cj(_~z;b9zL(%>ov89dTO&x@9%F&hG08Pko(%BAx<a)RQP2ASWJUj#tjCS)_%e)(@Ts;otfUPSRgnZD`Stp6c+=zLS0$m<9JCTma{15+@vZqSxhSt0EcWVc=%`(klL@K6@m+9>8o}J^Tc=4?E>916%?)T<7xy#}GN2t($lhKVj)CI|9Bv5aBU)%T%Ug6J*#?)~$3F4sYnJdyGGFcIQ5i3KpR9Y(d<X0=tpnPR*9<2!@Y?)>a!kT!SQOj_){_X8LKUGa`b)9cxeY8^!Dl7dOh2;RA({LtmH#ALBWa1QP|ogU7I)HlzZjQ35Q@_Yo|9(2x?S5XIBsXQ!v9<ENphgOW27C5=~7M~)2VhSeb>+#ihT$aII|HmRsTQyNLliAKboa!=6S#eHIfKW%|-XMwrcTKq9_#o!IJ3aP<|+#QO%(!8Hq`flw4CNnyFx^|g@&r<Mq7OjD)pd9cI8o8I@V!HvEimv8VMsJhm*pSo~Gjuwc>KDWZ!zwF*b8X;m+Gn^Wq@bR_8=|CR!?Y(r;7<D#Op4F6!B}HW+l9!YrG*ZpqvnCOPw^LJ#)-4pRN$oLpxs&&0=4q(4m~^7vB;tQ`-jQV%2@<ozx@l$2DjE1CMW&4L2{b6mww@n<^;`oDnxIn+1)XrW}Gd^r?Oq)W_o$@fp9b1#Ey_NT~02^>!kMzQ!NciGNbQy{W!u*X9Kli2s}mmgrnZY(kX6o)3{Nfy7pc3OOG5RG2p$|-?qz{<Vy!jA9r!C`!P~rd21sl-4D?k=^r{WV?vt6OnL{YjuThyb{zJo@6Sv>?fY=ukI9J5l}DCBk<m!LdrPZN@%1z2+g67AJ)BiPv}KgH>V}h5d!3q|O62<aXvFTo0#G}V%U*gYBJHPnTrIUJiH<gQTtLi0B;v=T$(?@I)HMR0b2407d<Pr_+S%UV|AQV1djU}7nJjQ@TA>sO`>`g;K164tM(hpWdux(SVWBsex6)Yb9Lyduf|5r3{+pxcN5SWNsRfq!w=S=uH)deSD)hZMuG)>%Y_$WelwWWESpDm_4Foogo}P>|U?!0s-m2a{Z`%jqK9|LQd~b{EqP(T_djXE=AiBkN<gSL}3<q`r0#`{dYC4U=OK&{U<Y3okGF6;oXIDa=-3mAv!Y;ShE@;5uhGjKequrvFVFe@MZ;%{`Z*_94^uw7J_p(>;jV!Vv?EUSDc!8KV-Mo;pl=y7*cpMj(E0q4^26eh5{fIgBbFR49<dck<k(T9p_f{!ByExW}<m_n9+0s!gnQ^10li%8-R(;VqYJI}CcGx6oXmyuKqyIEjT#b8Gd*{*?BRp>l^6>zSpeUb|*Ddo4A=Ep6L#|Z2X+e)afpY94ZFR_;)o>DbJd<L>P95noaTRG)XIpuVFe%RsYRglpz)6<nJEot~cG#VRevCFHrd`wXTqilRt;Y>p<Y`p{X{2X36h=t4&~*v(UKaVB{GmS?ogUM$r{uUj-KKdHVY@K96ZBf3+*8!wu+yJIkU{|U_Cj0wi5q!yx54YI`g^<1SDJwB0Khx(xMvt9{Sp2;ILB=xmYepE#C)4Y?)C~6B$@7P?{m#Z)&g@EWq8S~F(Kca5=z-sOf~0{Wr|vFC$Q#KGUfC4g``dOYWST1|E9OXqa(8Q1gAaqT0QMsDe+F2{JA&=6BXzIdBM*4E%r8a4gc<_Y@IJU@e)sfyKM*Ac@V&Jk7L3<aZJ**!!HTcdX;6H2qS&*u7`mIG>)CyW9lqnt8N`hRz$L)EvkDwE^w}7A`*mmn(5bEzS~s!22-U2yTdD~faFks!~6cc)k6Y~w7P%jU;p?s(jgQcLg4_f_;bdlGfX}3)Y+h9%+09~u_Xc?;A9J|m_Z2-7x>b3r?4?_A(+x0R7a8}YQ>^?y~r=~rmy9CpGj*tyxeSU4&C1X;J*oo;@9hRD90YS0sj2WSDGo&I-a<JnIIpWn>e=)GCLh}zD%-oD><1gu$smVlfZrl2(PbFykF4sKbbBmI&=JhV|@&JhCTBriF^~j3_xa}pQEp;%Pq<yKhjIR&<nrAT<s_Y;z2!!DWO{<lVnlOlO%R(qPmf!QU!6#)Ia=tc@Yjs+Me(woLAXuGcldK{rSfq|NQ3d+r3S#Is5wjs>CUnM9#VSz@+oSs~oar3If{1JlfZ2RhEql+5W<0)U580uHc*yA6o_XL?2T+95VfcFRs(e%T@LQ@e0i1#S-WVtHgU7MG6lc%$yF+_M^of^|%PfyVwxd6wx^lp9~k{RA6tzoLoW%NH#I8{AzE#Mb-?hXRXwiEWFwR26gRgqgdC&?RrP$mzpYf7xkrgLvh^hgZ`7V^S#wk`W%WIaz6l{9RNUmkB8`c{B#5o0EaQSe-wxW$G55%iqp?uADBtrGKaAD?EvHK{PX}aWcR`Q-RFFREuCE^4G3NP36T65-lH0f+$arp+z0E3Vt!-|)|rN=(D}Tc%PTu4XG9cRpl%)EKV!?Vf4~kmpp131tzyX$a=fl|+_NoeLA`R;_qL?A5Sg6Llz#fRsFIW_S*BaBVY5zfko#{rbor+X2=&M*Pw{Zb!sWwc`aQfAGg<5qYfQslDtIG@)9~AUk#y3k^hI#Kf3%iff-fiMM?j+l*@359;)~$yVBe;&%0ip6&QZ_uXZ{RDY3hN4wc*#9xz^IYw2?0O2-(6O0JtT+iYNdUhhu`}P=j91p1&~fWcS(yj2oNb!_)@nS$_pn%yD0a$FD#6<mEolt)DWsL#X38C)?vQuPeUJ4W)f(o?7qtF*7WFI0;Unj>r`8f{hMLGRT`3=|#T6lRs0(PByB1AQ!|fIx~van&5@5@nok%?}qBU@?QqP%J&hyBLf>%ss+&b=?oJhZ5tB1nFAv7kmcMsN;)b(zN3QgSBua|dJd9>N332c3%&7&&2no(hwOc(GvUkX(A%oRZT0YJ<711V4QgU*#a*xxJ2*b)vDaMiyk{Ps1+PzKJ7EbHYLQo?{^kMbFYX7=?NQUtYoccyR~?xx1g}CXS2qNy)*C1XUStE0JO*8kq+_{}%=0a6m&&8q>Nh0#BW);{@D~OQ|Ka5H(!`M&jh#C<04(FJ>-kF)1n|U;S!u-<GmZ+p6hJd|-r*@1{N20V?Q&x%fbP~CShFUW5@d6bTJDWk19qV{W2aY}r8pj~ud#bTG2V199(@F`Z)cUtl*ZZ)8z13rO)H<Zs+y9<U|wTq-(6(g#L8AlqtXU(q|G?p#UtwUR}&8gVRz1>5#H_1sfoD3y}OztPZ_p?`~i*G7eI!E4M1flx&)sF-UTV7c8*=jb{&5?dZR7g7&?g!dK^)P5W}8SrWe=Yj>?+{bY)vZj=px&CFe3X&UD<fAGom@oJ&ti@vlMPMx&0~+oYJzHFk3B>`DB{ZQhNG9Cx=@ZL0&X|Fjj#;YD%u?9MU(6ByDGxZKNQe8=B$(BpVbE`3IVbZ?J=HA5VeXtutu@g|O&DPI2T^8?3p>^OcnHt!$NOGB*5txPYDJEK;|?wk^yO`cpF(&Bm$M4IJ?cwtP(X9Xw}Xtav(B1d6|QlNE{W80E@q>Jk+-NLGF9aD)HUn^1iQtCV)xQ`?B-SqF7V#nZaHDLZuot0lnV`941;LXmiJO@&3qU}9O1S3wPjAHgTA2R~srxP^8%ilggzfy^m;aB{lClFXeWw;psz}gbE1I+!|GlwU+{=WR>(Qk8)mo0wCnpv<%Ay9C3m(90LO83Hq!{3D*y+DMHK(y?3;)6)&ihb85wb)OBHl0V(<sg{fF35T{f-g}kBEqa0<0>Fe24;@YAHZE6T2fHk63G%hY4MMB`i|lTK_X6+_OPKJi@VP+d(IG!?aD&l+O%<CHIr}rgk#Ys`lrv!)he4iN4e^Ii!|bdpB)T6<8k6*G!D8vQFJQD_pge!Z|wLer)hu20o5M9sBxc8oqmN%hkFL)PIT;^161WL`*@evVeGnrnXViTn&zu5-@QA<Qaz<rA@iD@`Iz?vIJDFYlBIdPqCq<%M`jDTN^A$Gt?fG|2l5m34flx05tV2K5x?i*^pp0%?v00U&*^`*N;mbch;e&+x~Lu|LP*>@`(3{edVI?@(w(ao@m8L0$aC9QngQ>*tx&70dq+2tHg16?Ba#E^?P_H&zUVO+nM}mAmUedQ#_AsEv2)!<yRCNY?2+vmJJ3-rekrSSN2`9oFrsF);>(k`m|UOK7hD|g1s4`4E&5gm$PRigm28oVw+EqG<4zwNXzV0)tM4^i(mHV*;9s~20XVcCGfh@x_2Ar<9TTRgQuc(a*P_XWP+|w}0zrV^{O6BvzW(XWH_2Due}53x^V64V{0mH%*0ujVp+&BYv}0DV8x1pyEnn~QfPLsqvCc)lfaV-vDdz#^21pxpDY7wlK*4Kv^mlYR;7a*2oum2|UY?#Oc#4+DAuJsxlFxAh;}@0eY|x_!I{F3Ahta~+>d{pzRE(>Q9OvpOigJpHR+<wZC1PI0k2R=p+%xw(GR`>Hq}r?jEutPAcVXv_4kXh>;^!K0meKB;KK%!~IyjsNz`AuM6wT3d7oOp5yLw?-H~DWLU6qv5P{96HP=YC%qiz-fSE){5LuBB3)a!l#w6h~LYF`<qzA()}zpm^KL<yaUjCXJ+rGY#dI9juP*>76xA4`Zv`^rcbwrOK*Kei!?I;IiUNW^4z=?zFW+ZFjNYdquBD>dtv40m+R+4d<jnrDD<@kvp?37yd(N-wkJ(jkYm0Jcp+Q|&yZ+aD|U0Xn}VpfKf2fV-QNq79hCO%vgH9SpQNZr`}wfw1sMgP<F}+uE%!Ujj3F_KZ1@+lcx*jjiD6*isIPn^*}ZIiL4N(0UsDPOWL6T)$uB?Fk|_BbSPr8H3ZOSgvedn`c<SE-3R}AI%>5_TZwKoX6gg?&-<N32dPna7sI9lHkckl7xDTFz$VC?}6smZ|?FYqH_;u=KlduO9KQH000080000X07nxS*&qf00OCOa01N;C0A^uxbaG{9E^2dcZmpclZX`DnMfd)UT5G}JGuXdk7(zVVg6)<esSOMo{qMu-lrhUu19u@ouRKLE7?(F9v-IsIFK<sN_ICSpOL={JeELg%kH?qz{pt1X<v+KFn>~O1^7eZB^zC;2%07Sn^2hnUmR8Ec?JZu<wf%hYB2Qm#pJ?xg+doep&)464f4IGbJ-_(*_SbK>f3K$dwJvY+8uyp(UFTyVY939`i@tuAzvEnR|FhX2>X`SB-_9@o{`2X6!H<{y{_uZ*T08fV2;fwe$1-&CD3B4p16@lh;imjEYxpB#&_(<h(FWIa0}@@=NC_N~s;9R7vA}$ccs`%&97r6ibE&{s6I(bF1BpIb;RamC+Ux8|6-b<mI!6Z*rAFok=1k2-X%$FpO~;;!BzE$wNTq1PlcF>rQMJuofMe{%jyetj5@}eG1|*J_d(;8UnK4Mt*@YbA>cwjZs@dQr6cst^q~X*CjyNYrF<VGt?z$`RoLOa-U9$j*B%KygA6NHdD=N?dENP{>AZHz8tW`FS$c0Z{`veln&Sl70^EyU%8t`;wvpLEH`bR$ctsCi5ICq#RaMru5W5hR5&3517@xkKbW5iMF4(J%ys^eH^AaPF`!+}b%HMnR&j!{#l6yWJ9m8cd2Mn<Y?X9b?uBFS0;IcrTm+Xq<BXtjO{5TN23jr4F}Wb9;N=>gPmTZ;_3!;PcLs6c0Lds+%1XMKv9HUqeOEIGp`FlW4O+oHhT<EPZ@xq<8GBbGC=m*Q)5kpXm4zK##yjwEU`?h6@f&*L8j=<Gd2#Wt{>iOPH`VxT@o-C1ZMiH+(cq&JH;lxP7X6Juo#=sM~NO-Z0@&dRGS&=`Ndzy3kuPc+y+Mx52y8>V8vh^wEnuwl-^cVg5|4Rb17L>`)RAY*+u;uTaYTtr(b3*`1LVk|3OfU!QxGlvyS^>Gnf(jK7N^&*a5nr@5_zldI(8)FprB8H}N!6@!U%)wR@xWnsR98Xb$B=V@J`!g5uoP}Z?xO-}o+z~(xuV+7+gB<BaoPA1yx~}g;4yp|~My72-z?`YvG<O1HeKt=PH{i+Ds_RiSj7zzQZ7C_}ZpuYm#{rlEv#<$q-=OZyJ5e(T)Iqw4Hs&S;>SJ>qJt{C~TDnZFn0n?1v9}Cq8+gw2)ive>=5=SLW2Jz3eTr-9#+YjOBIe!(jdjoT<^t57F?5}S3+T<puvNGt$GCKv5ps;ueYXjW%s7>1Ulpi5$1%6fpgzSrk!xpASK%Uh#uf)=;Yi*Esy&n3DUhytQrM9*U}R>iBn{NtyNG@5-Uf2kEA*6wz`UL&yxIb0;dCvrL0ztkn4wfa_lEx^h8ARZW|a|E4Y+$2u~teXu}_{pf%>@S{Ye)B)-&sDpT&lZ40kz1<T~=?-2_-GEZ%v%6wui(mZiRtk&zrdLHDmOq7H9bk?ZKK3<f=qdl8eDHU~zAi&H%&07$Guosj)-UX^zc@SNe3)kn~PJ*oS1DT^G~@7c5mwqoCtTZ8r=_uP{h*)80QQU>U*{Y8{cUMf&=chuG!^8U=3@val}6v;*0MT%~qn#swTP7UZE#e8uUpqfb>-U8iYe<wyPE<k55NqyS}W}&1~rv$2*l0;hy%o(kvn;?75+Uk}BJrj2k=RWdCz${eVSqri=tg5T90Z#?1%Tx+53$0bonn0zn)r6dYd2MbP19`XEvW-d|sJNCnc!J(Z@`LClt+4{t%t=fqQhU6fq_F~A$9scx4bZy*FX9L}g(h+xXILoG0r)uzdm??fhx*b4YLDkUna#i~3_6?922jmnMDLrCMD=XifG1aI<_u&XH+;9XkQyF-jL(3^I@HDzU=~I$?YV#zd(zTK1*%z6dMspxbSz3=96-01{Wx(V<g7XK{sb+MsK%Q?@4vo?HfTChaSt)u#*UF2H_$Vh7cu54?WgW(-hX1B<1cyqCthBkp6~xK;r!_N>x=#GpDsKe+PV5%Rp~C5$tr$csQ1;qmpR;pm+&^JA=B*s^x&zi`YEJ$+dD<(=1S)*`2Nd(0Z>Z;0u%!j000080000X0HbuaK<YdI08!@v01E&B00000000000Hgr`0001AUu|J&ZeL$6aCuNm0Rj{Q6aWAK2mk;8Apl1c7ug^N0081a000aC0000000000005)`6FmR`W?^%5a%E>OYIARHP)h{{000000ssO4a{vGUT|)o>000"
REQ_DISTS = {
    "donfig": "0.8.1.post1",
    "google_crc32c": "1.8.0",
    "numcodecs": "0.15.1",
    "packaging": "26.2",
    "zarr": "3.2.1",
}
DEADLINE_SECONDS = 120 * 60
PREFLIGHT_CODE = (
    "import importlib.metadata as m, scipy, torch, zarr; "
    "from scipy.spatial import cKDTree; "
    "assert callable(cKDTree); "
    "assert torch.cuda.is_available() and torch.cuda.device_count() == 2; "
    "assert m.version('zarr') == '3.2.1'; "
    "print('torch=' + torch.__version__, 'zarr=' + zarr.__version__, "
    "'scipy=' + scipy.__version__, 'gpu0=' + torch.cuda.get_device_name(0), "
    "'gpu1=' + torch.cuda.get_device_name(1))"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_checkpoint(root: Path) -> Path:
    candidates = []
    searched = []
    for source in sorted(root.iterdir()):
        if not source.is_dir() or "biohub-cell-tracking-during-development" in source.name:
            continue
        searched.append(source.name)
        for name in ("checkpoint.pt", "t03-v6.pt"):
            candidates.extend(source.rglob(name))
    hits = [path for path in candidates if path.stat().st_size == CKPT_BYTES and sha256(path) == CKPT_SHA]
    if len(hits) != 1:
        raise RuntimeError(f"expected one exact V6 checkpoint outside the competition tree, found {len(hits)}; mounts={searched}")
    return hits[0]


def exact_dependencies(checkpoint: Path) -> Path:
    candidates = [path for path in checkpoint.parent.rglob("deps-v1") if path.is_dir()]
    valid = []
    for path in candidates:
        if all((path / f"{name}-{version}.dist-info").is_dir() for name, version in REQ_DISTS.items()):
            valid.append(path)
    if len(valid) != 1:
        raise RuntimeError(f"expected one exact offline dependency tree beside {checkpoint}, found {len(valid)}")
    return valid[0]


def competition_root(root: Path) -> Path:
    candidates = (
        root / "competitions" / "biohub-cell-tracking-during-development",
        root / "biohub-cell-tracking-during-development",
    )
    valid = [path for path in candidates if (path / "train").is_dir() and (path / "test").is_dir()]
    if len(valid) != 1:
        raise RuntimeError(f"expected one competition image root, found {len(valid)}")
    return valid[0]


def _remaining(deadline_epoch: float) -> float:
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        raise RuntimeError("120-minute wrapper deadline exceeded")
    return remaining


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_deadlined(command: list[str], environment: dict[str, str], deadline_epoch: float) -> int:
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        return_code = process.wait(timeout=_remaining(deadline_epoch))
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        _terminate_group(process)
        raise RuntimeError("120-minute wrapper deadline exceeded; inference workers terminated") from exc
    if time.time() >= deadline_epoch:
        _terminate_group(process)
        raise RuntimeError("120-minute wrapper deadline exceeded")
    if return_code != 0:
        _terminate_group(process)
    return return_code


def main() -> int:
    started = time.time()
    deadline_epoch = started + DEADLINE_SECONDS
    input_root = Path("/kaggle/input")
    checkpoint = exact_checkpoint(input_root)
    dependencies = exact_dependencies(checkpoint)
    data = competition_root(input_root)
    package = Path("/kaggle/working/fastsrc.zip")
    package.write_bytes(base64.b85decode(EMBED.encode("ascii")))
    if package.stat().st_size >= 1_000_000 or sha256(package) != PACKAGE_SHA:
        raise RuntimeError("embedded fast runtime package identity mismatch or source limit exceeded")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(dependencies)
    try:
        probe = subprocess.run(
            [sys.executable, "-B", "-c", PREFLIGHT_CODE],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=_remaining(deadline_epoch),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("120-minute wrapper deadline exceeded during preflight") from exc
    if probe.returncode != 0:
        raise RuntimeError(f"manual offline dependency/GPU preflight failed: {probe.stderr[-2000:]}")
    output = Path("/kaggle/working/fastout")
    command = [
        sys.executable, "-B", str(package),
        "--mode", "benchmark",
        "--root", str(data),
        "--checkpoint", str(checkpoint),
        "--out", str(output),
        "--gpus", "0,1",
        "--precision", "auto",
        "--batch", "4",
        "--det-threshold", "0.97",
        "--nms-um", "3.0",
        "--max-points", "512",
        "--edge-radius", "8.0",
        "--neighbor-cap", "4",
        "--edge-threshold", "0.2",
        "--max-candidates", "4096",
        "--deadline-min", str(_remaining(deadline_epoch) / 60),
    ]
    preflight = {
        "artifact": "fast203_preflight",
        "schema_version": 1,
        "checkpoint": {"path": str(checkpoint), "bytes": CKPT_BYTES, "sha256": CKPT_SHA},
        "dependencies": {"path": str(dependencies), "distributions": REQ_DISTS},
        "competition_root": str(data),
        "package": {"path": str(package), "bytes": package.stat().st_size, "sha256": sha256(package)},
        "gpu_probe": probe.stdout.strip(),
        "command": command,
        "started_epoch": started,
    }
    Path("/kaggle/working/preflight.json").write_text(
        json.dumps(preflight, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    try:
        return_code = _run_deadlined(command, environment, deadline_epoch)
    except RuntimeError:
        (output / "stats.json").unlink(missing_ok=True)
        raise
    if return_code != 0:
        raise RuntimeError(f"fast203 benchmark failed with exit code {return_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
