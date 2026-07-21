# Private T-03 first-candidate training entrypoint.  BSD-3-Clause attribution:
# TemporalUNet3D below is the released implementation from royerlab/
# kaggle-cell-tracking-competition (commit 016845f), retained under its BSD-3-Clause license.
from __future__ import annotations
import argparse, base64, hashlib, importlib, importlib.metadata, json, math, os, random, re, shutil, sys, tempfile, time, zipfile, zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

def _canon(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

EMBEDDED = {'registry_projection': ('c-p01!H%9cj=b+<^f>_}k)p`F%3@%U6q)s65(l<3z+y1(zLc|ry-Sxy8|ePKK8a*iQPk&ue(2ZlpHuz&$1fjW|NQ;?=ilPzukq{WZ{NTE&(D8+{qo=8-#>r({pWx5kDoq%{rct4KfZnZ@}K|w@GrlA*U!Iy`}BK#^R>SF^jrL9@%7`E)W3g=|NGfbpLc!kuWkI?`ZnHzrV0@)g5iAPw?F>=`TP0Aj~^fM@wcBo{_%Oomrwqsz<m7D|NZaJ`~QFH7c$dNv9sf3`1sque!9T3e6=PEk-LWPm}GBwKs%%OxWvQL2{>l0&k$3jM%Q={jXrsD#;&Kj!?RJT_imCL;KpmsH#|fsL&m<v6U$%#VYt(>dbD`kubz{ABwgc?Ho!2|cFrX$_8XqXwbY&X5>KvPxfSUPW=3C-^0r@n>@<*mho{4oy*F?l+Sq-re2nK9%-Ec6@obzs^lVpUyWw7s`92sdC2!Y<)Yiz(o2Wq!64`I-v$|qml9zb6<w#-eA~;~up8SS~SFI8`UgD8vY{Q#3!Y&?Qk9@J(3?DAHcxENn5?@G=fIA#s^Br~ZEuPWSRoHOpu0mg%Kj$N#cFOd5S$_s+!e9!@R<umi;}Or&oS0SR@9@aLom<kvPm!i3<B?C->TV5h@n8jWI3sai4OLX%)`!<%i4M1TIBE4F(zeGQ#i)R9coH?ZcfZ7AAzsyIb}I5pLLKc<kBw~|@)}RYv2|^_SI;q4L_X$MSh1kaOFY|DtYoUgsP#$fMsIjFg@@UtJ{%=2*ESlPrR1Tm^zoSQw5{slC7%{|^$4ig5#8F>s_b~=ubgTE_js0<=?O0pHN2i{Kh{IailAoKcwz=n)gMi6Tl?Vk%3oAlub25%QH<P>a09eXL>u#Mze+414x(E;r7{d*O`gRX1WoAG50**kO|S7lGwZE*9;`UN9O2vg2v%mqZTu9so3v?-t+}#OYwZosp0TQU+~PTU9((PR@4cd^86V?$+F)GccE1XNW^LB$C=J$v_NX_?G|_Byn_oO*OdZ@e*??|y#G@aat1@<6@&zbMg^iD1%RDGI!J~gK|HN3f?iSBh9<9;b;Ic;jiy!@9TTZHZizj+i5J?9*Jq2hiZ+MIZ%6{MO*U9jzL1p1m&{hTaqhBu8Dbg8ui)Z~<nc)h(0-G89aeYj3G{4l(r8Km+so4;5We23kc%F(|fZ8n{;9@V6-gO9|gg)wxs%9uIxBC@klB=Xn*=?wm`oY(JOYOTl-tt9`U6Rn3%cP=<z3`F00Bj1`WxiuO3GbB@p!F#<`zxPLUfN?^_Mfq@-BwiW#0CPa^m^s@#nEJ6_BRl?lryce%>JX@!j6yqLzvd<DKGKNxoe}qSj+ANC-bX6t!(WEmwE%2Q844|P{w&wSi@_-ch4f7Tfc*BiBrIuWK~4q(k^f7a~{gH@hu)7C0780(4BYUdew(JqZIV*`smOJws(o=(j6AMUi)K6Oq*`w$5AIGhL*$NR@>Dd`%%(mKE(6JvR)$Pn<HzHLSN5!=JeHy6EE?EmMfjjCl(Jzc;Mst7k3}E;_VjCZrvyK@;r^gx58sR6iX$&^`{s~rFzbBuIhr3&>!PD%~^O~`W?(EGKYBi!4@`jqd)RR85M4MjmKxJWkFuj10D3W9%71kjLZ1>k(%Y;nyt>wTP%IVL%0NCUh?TIG>A%)#Ol2)kG!7e?zXu^@Fkv^)GVw1tUww?_WbHky%f;UOa7XK-EnJ3VD#y-XTJ7Vk3Q_{7LWCouuC3tOrAFNu^#TK85PsF@w|>aXvvjX*g-n-W4}c$dCY#hUossG-DYTs1e@C*_X~0>m4Uj&qvb4O)t(KblA2Q=&zD80mk7Q*kDCU&@4dO=xRuex^6`)77qCl1!DWAAZiH4TT84>aSxH2C>?fdV$D*)1JcO7lE3s(C=)I)Sem!6AY@_Ebp0!KRjoA>}$W~#UkM%I4+#KBU7ugD)r_&oWb}{|VM?Fs!jy>8Xe^GSNL@UpfUCQo|Joe`S`_3^h`w5hDl@V-v!z{gu`aF+hUE?&A?(nQ$;(!G9DtDSXa?i*3X+=6o^YVOW0WB)#@`WW1E7#T>ukjOZL6`M;Ek0eb%}A+>g=*^`>tR<n%DlxRgN!Rf(`CW{h{yVa%$@qU^vkEWKl<^g>&US)YWJA$(5sH^+kS7Uc;RyZdzvm4*{$5hTfVn#Yo=cEDWeR)fK<p@93bwG@ti$(RoBb<%(InNg(^Ty<>WG+{a{I!4K^<Q7gOunGi~G;$<gf9@0i(Y?3aF-*J=_u=akbA&pyYa-k9D%ST5@i^FW$Rb2irLRqR1t=OMdzNuXOi7K&xtw@Euw3Jo65-`m1kbGXcRR%|jBZ?Y2M43O(_o@5);bFE9g%Fb1J0)37Oq@-?3U;Fd3B<sB7ciw$$NWx)Sg>`$A$NH=hvAA8vvvh{m5<z9oUd5=(E1#B=E4zG)XH4};yS_}ZfvwNS`8pe|Op{;wcY*ANV%AnAY5{An@iQSyhR@6MCc$Z4GB@s7>r|x5^m_izvmT9a@z^#iN6DuGRjIf4eqA1mdRc#jcnP&1-CkWX(_iO%dqqGgyGuPLI>WIHysj!hv({=n>Yt$_GH>VAMXP@0K9iyw+zVSi&bN4p%!ar6M>MDuBJmb77-roc`F$rX(A)b>Vi0+T0R%A{Lg;wZ^X)W%hTP`+F2;g#rdb3w($9^M*ZD(Tw&k`SW{*i8tDJIcW}N-C{(z>GL%74EEuAf6f`@wPAht(6AGQ<DRQDa8)61&WRU5tg%m%OieGro7Er01CHQ~j{u+CUhzUpUKXDrt)`9eXPOzhEuC_UI8`cZ$~-@)DdOzGX?0bE$DRb*kAh|-_ujZe$}vK}h-3Z9VKHaHbp_;sGY@&jt`_br~XClN(oqyoHE__ZHid?9gL4;2%1Qui{3F8g#5AJ?Y{`m}Iv>(44MqSl(OD92pH@!Fr~(`UQAucy*-m7ddpHablzY4^BarF@aE_scdpG@^xVEy^;5kNH)Gpn=I{KcW77^V;%~V^F1e%FBJmzw~_exjW2TJj<!7piC{zOTL;t?w9w{cd^GUp89>*wo0y20FaKy{>G{#oqXG`vkK$QeZdywu;f^;^HJN6T!33VQp)+q)#ci%F__jPpC;#N@?Y}1VHCd~2{am^g17NFkG+zY!M@~on{zeMF}o4Cit?*IY_YtR?Y_iQX!zW8@QQ4L;tYQDU*#>R<jZaREX&%n;?y(#<7;T!ul=7QjCFe-#7fFJ)$&Npai-Rzzuu>A=g#P@{xOAaFu4V`ElmmVu|7M;(0aC8JY;Pu%)QY&;vV~T-gtJM+iv;GvuPW1ua1dT>LVZXs|p7$j>~>#)f?(m+0|u48x>5i^Yd(rJKxTSdz+@`{SwMh+BJJI*sCARq*!!|Crm_`nGmE9(d+&0K8R+0>?NL2=C+6eTNRCstJ3{(eMadq-?#Hf4n2a}tj%gB{oKQNoIe~?slkD_c-EYX6RRSMY&CwJXFKk)yzVU?Bw_Vp3YPL06PG^DtAiO(xr9qRt1;QRUQo(;@6(~T*Y^<#td*Dk)JtIu)2ZU4xlKuWoM)4gzOyR*Egr8eb1DYI1*^AGWsmdEjnCN5ZT%_TjJ;TkTUm@seQ1yQRixuhZ|BvI`j%acQMQScv#h~uf1@vpw9E5u4;(42N7fQ=Eib=b?|atQG3hp*W9jDQr)qTEt;y^8vM0>Z@p^q!`CK1sn5~T!Blqk3h$LIT=F>bUO?n!cL?!Lq<Z)gUAwxkf_iK05-jdGoa;M>5>H5g;B{+=RH6Gd6dat-4X5G!#qrdP4+IPS77aJzoh+aWBIc7PqS3WHogmcIL7EfZ6pn?-W-ghyPU;C>wD|{~VYvayZp_r$Qn4`SqYdw_JQ|Ik@+^))%G9br-?qW~#KY#hbdbTsfbBSk_a-K@5EyR*x&&PP4%a(h;)X!UJaqhBSWl}1RRA2RcWoYWUJ%8Uiyw&rx7)-rLgZFwpZ{p|!ZsW%%mE;-VG<rEp!-szOUwkoF=Dh8PM|aZ5Ja4vX%W^f%*Z12<xEfyL;VNH3+knqCj0}9`_dN4xmwNsT&F9=$KoIsBrT0gE@9eR!=SgBoKT>rKpsE|H(;v?ZTCp;3xa=PWb0IJ)Yl_2GafUwjdxaR_`o2||Z1ZAxTGmpvd%Vuq4Ji|4m*)i$V|h45KO=TzJzx9Jw8C~<aEr$cvT_}xoZ(uV{PBGGT>jew?G{fE@6m?CTpglyc-$`?X2`eal~oA)DC_R6jG&f5E&KCWe_(KqTmPQj7fNNjof>BDupa#~HpiLs(x0B2B;c#oXAbNuy5M8|*%i)3JlA;kURt|#UIYTmd&S~ozhwrZ@MV2I_myM#Y1R>^_1*As9@2$8%R}Ab8QeSZvCAF1u9x$jzv>T+u5dfwn{yPELjI_!v&}vEF@AVV+x=Gm<QkMx_Ujybqh(|~o-glEQ@@>`&sFLvky~g8hE=9M^2KiHz;=yCeQt)MTC^FE;p2G<dGQXc+j!<8jDSupEltMk`}w{D59Hf^OIec!xb}H#SY-xI;n(xHDzQLuTYqYcNXs<0RTgy<f#K_Yhp0*q;}(x+6Rq;I>qE|~tzX~&(|q!|7;uTlLC20&AqeFAydL|JPX`Xd?6>RV<&+ya61SdFwJ09-m};_B`MAWhsWH3m!|WUa@j9<Qlcinv4>L5a^b?xIh^ccoeSKe3AJl*N)B3G%U;g~+@#)|HHN~fI|D|J6{qfPylxr=e?pAq?3pbv(`t7P-%Hldg$^BdfLh*_L6A&rKLBgX~(f-$e{4d8r)x!', 'ec92db4d5fd09e82e867bbf32c1d0d8100dd55a9cbea81c62263ddf03598c115', 25349), 'split': ('c-rM%+in~=l6~LL(6f7+_ltjJ(J1gL7e<y0U5swb_}}-iq)e%zB6FOjYQeMnB0waQcSghso?!j+D&0Tsdn%7tpRca(@1DN3y3_scrtZ>H-9IX^`!26OT^;_p`uxx5GP`|!xcdA*SJy)QCZtL}7TrS6(w8KS^!VVArB!IYiKc_HF?yf0nzF1CzR7MxFTu!?%WIaC!gQB%aoLFMeJ*jz@?D{(W~^{Egr-}SFIh4<U%QdbJK=gVwuY%f<wXqL2km>-RjD3AeaX@ypb@8BDXH4jg)rtN%T&}+rHR=`n@x<;OkY`)#&D{Y#%EWX)M6U4?HH^HCJNEIlNBAwFIis2($E1TW$j6aEZUqEh0j{(V518`3Fmc~vP$*EG)!I1PUT*^n7R>csZ}K>sw&bOom3JpS+T~JL^B%A7*VCRX39!7INh))=!GdUCa?5MR(2uFTsuaPqLe!4(qFQ=j@1^6Rp~{_h7awWh1pO-df_n1LfWRZzxw0T)zj_c-BbC}>ec7R`=|El>ifR7yIr|M#?@yfri7Qh^mw24x1Wv4Zl}7Iy{OT^n^fmQ6Q%g$R7{Le^)T=^sWG=8vM|S~NlB?hQOBvZc8jf9k5jt}UijkXRF`wwX-q+snWolkO;h4H)!OV^%k4PTRjhl=<~SAWue_|8q`K&2=%lrnjI2V%<J1&I5;+!<iZf|s@5iZGtC+eWeW)tOUIeM3b)QZ5WE8zv=_+RDj~}0|4lDXFgu_8f><-KN@VvzBW`DiUH*I(Lcz2K<$db__@6U4cR3UF}?{1&7@9uYB_kSW~DrY`j-M6pEs_lNRS9*(9YkEu%?QwVCuJ;d*_kZmkz9eNG#F&y@XWc_i6%xo8kE1Lc4s!}Vg^1nS5)QogLYr8OayjZ4drB?_CG9wQLjWE|B*vePK{0Z#=1`-_DIJbe3wn}80w>ey&@8f4%*j9`O<A?!q$}-AE5-TTTWzmdQ7DbsPI+v}=-YJCMfVczv|hZ1gppqNNhIQE$6Bin5W^yrVtt?5!X)Dp$YyVx?%LE=rjw2-f(V0bz`pM|wlY~eWkqFbud(M=MqiwWvmnhamXIXY9)0M}BqipoE@YNTr0OMvG<N<ZA5_gL=GrVoJifw?%u^3lkFr6EOCJ-?u=iet*DPN!$E`qUs}hQ=Qx8#%3Bov2wHDGjS52R@j21adf$?dqKNP{#7KCe4HmM-1%Sy{^)im=OyCPxkl*}q9*DD0%oP}MZ#MnN%L=Q<5!kpD)(_Ti2)x?@XmrRSIixUD}$;ox>5NO3diFyn+NikH7!OrYb8#lK#REPE%qBp6Cz1h+<WFtmnC5Wn1blF)crkRJjtx%=-qE+p*3A)T#I$-vk7uaiu^$%s%Zq`=Rq;p8RKv+U2&R;>NwGfPz8q2MG!ReT?Lhi;!okfKBK=sbfUqSm=oog*=Xt3OppF3YnvaTCwB-sNru9fLk$$A@N#A>&WPg|Og1Bzp9tti%XX<94L`g2wl5|S9A4JWIWn)E!b4nxC2(~44|UsbB6%vr@k9&{6Y92!igx${<L{O5<YA%iFa>UPdThA|Kg-Gx*n7CKHnv<`-gsj0yrY0?&G$Y~#1i1U&~4`{_m4clS1Mr_1E$fji5QmWX7byl(#)`7%f4yZ^N$d;0(18XZbb}F?g<svMi65KQ{XCR-u4c1|UOUSQSr>|fVMH!XFUK6SdG}))lJ1D1MtA~B&RK=9lKG`r1!UroaAXQ-DG?QKh5q7)LF6s)YqIy!j+NnbCL(1rVFfhM-h^{L;XGI8E=u_QcH$eJfr_Yv;i~(9wj2(xn1y*k|Z=AqUfxJ+C*lbPYc~JsnH#Us577`OCiL`2(jS#T)4A#BoC}3J9p(hCtl7TGq0p}kQAZQJnaLOtH9l)lr!&rpjmi@FvLzSv(2s@RN#DI&4(?mrXv=AMZSF~1Ik>uR@2)j?l-m5v7O&DC5x&af<M<)(@7nZZdww%98wqZv>Q(^xITXR;gS=HNKht!26DkwHDWlYUs79F&agRF%3E7NSmtXm)QBw~{Yqah~rY$9Ti8tdCJW(6Y-s$(({73{niJfwn#B`R~Dr#ESwHE*&N!z%8O0&&uM*yz#YFyLc7`Uqt@+ngrI*cyiAQq%#GXSTqT={R9d7OW{wkQQCeUwNd*l!d?!#1_CPrq09K>E5c48mvF*tb<sej94W_!8{JTR@(;c=<}QfgAPIvdJ!VS6)*DZS4CRbanL~CHLYDzlhDh<zzop2IFO-b>_ji-tcc|}XlvA^RB(QpsBA5E5-bvEI}Kf%f}A!$9(bhU=oPf;G=?3fRn7rO$RQ^-tA_f3l|E%9m!UT_>=*+%WU8ujH%f=`r*N7fpNxcsFxvn{*qnw@kpd)9fWC2Q&T2Wv7(#GH!>A0dP2IrR)*0#@QX_eYX!K(KN_%U8YGA+M<C-##&6MQ_ni|>$bB^<;AWx?(j34L6CWs9I(T5|Z=B!eTE6zftd#F?B+DVuf%s6aa%)Y>qp+{Ow(+h0?ogrlFs=Au6PyE~sNbec|ny^iQ*@Sr2Uj-r;QrrxH2?dQ^)u-tN(t}>^ZU_MB6a=gYb5?`_DJ3exL55LT{QQ-+5GBB|+FICqSj>4gATuCtnnO-I*f!S86<Sw&RJto1z!LKt$yPm<!@_Jx10~@6RXewpQ~@HvvSFVA@4@<<4s65m(O5))3mt~m>1~?Bp%OZ@2hxnLLs68e!Uo|xW)=s<VYjAa%XAzd4DnAMSO;<s@;{|%=CNsNE|?EkuL4H1ENvQBYdByHM@$Yu>M{($*Q^4Q9H+!IYz6PDD|6>tgXmI16>U9i4k2go+l7}3o2dw_e8T21YDS$BW-SadSTN3ZC`9e2Aw%(W4f7s*U%@(R&{=ZM!nsSRu3D3E@+7eHlm&4DE2&59RfsoR?JQDYL&FS(&DkK{Ov|?0xrb0*kQb2UJ@*V%E9*2-eH=@`Ym))_b<jZ0(<=7{gD`Yua%>7P!B&w5WbPAsVn8trp)l0XF-#Q@S+=KkoC=KvtN9?A-=^X(5E22p5_V4zDwEVu8iZaTsnx>D)udhpK-GIBa2-aqlTGO3R3nPP^d*uC>q$Vh9H&C3VrDW;g_%|{b0ihePC`kLR9FVX`9WL;Vqlt1Tn4_`A;sfyDx|=$jT}jp1rteY2pAWjB$=H7?M;ZJXVE~Rw@k}g;QDq_7D@?*n4vBXU^Cz?>f*>5Ryvs-m`DLtkky3tF+jg$Y|w)ppq%sprnAD?B_qZmljX$CQh+EMj4a}2QHPU7oi7lJ5Im8l!rY6Mre+O;&Xal}4Fh(Sqp1;q)YDWg6>K|0Q(>z~PtFCv129o1?hpah5UTbh)tli>&?Gg$LV|5VQe$)wYKo*bh^s0{z38pO$so=bsXGkKH~Z{4wTc=5AW13=mK;PSsSse$@uX*J_`6c-G&MjnzS%p^sTOlyBXM~t7n)R>q-s?&`b=h01L+RfMN<91`kAz*J5>C$E&L{RDEQ_a8L<hlw4@@lgG~md6o}Kz^cV}{xEq#?A**RUaD0(uEno^&m<6;3XqXtHrKtguz>`^bz)X;nWQCmZKw6fj`j)XtNY7k{AQ4@hqykoDG7rGFFpxW$2X9<()seOEZpc(g_M``<@}bZgLhP9LKvR>?HPRYj<=c?ROk$Htg$W|5%H}Gm8y29^p6(q0E3HpX05q%~P9fqLwF9VRnwl{>smZJ-SSUUSOI@qdK%$Y>vIHdYnamChDLrr$NzE{pLm{IW9v9eyWG0JNuopV%S@Al_D#;p`3j5DeX$0HGDrpK$g(z^AocRj5r$Z(qt}>PEs57~0z&fkR*eU=RwhpbqwbC-3^%@jVAYEf<M<BOOW~X+CF$XfXCef{8$(%LY08ax=1!{GTp2CJx>^+^|R<X$Tq_viTl@l3Thkk_GA}bG7ID91j#C<+4R0>Un{MMfIp*IKB0vQGS@JqX?ySpdO8|@*pMDzwoJGD}QMe^)dHo!+xW}MKbAzS_InZs5$w!#=qx#F8_Ua_)tI1Iqtp*kmkR2;zGlm)d2@D-Ab(I_CGI-A2XiVOi$L;;GEkga+~*x0`vo7gqjO_)>%<M7qxZ172ga(BSl!T>F&IV)hrB{l+#zN&yZW>=qrjp4_oSPc=wO1_ubQH}lI0PrEII^;@%VDYaWVi~gmiVd~`hH4RK53ve40xye@fi}YM;Z6p-9usBc0H6c7!|E|}h4=@upckN%D6omNP_rA@AAG|K*hMOvLeORQ^>%G6R9dXVgCz$ZnX&*<Jg|r_(PE>@>gT&Vpg3p{p-poJ2(@;Y9kWlgJD~e<8<CRN;|gt>UTeD(8Gy6<W}LD$g;y8m&>Xr(82b$~lI`qrwlGk2Y984N$lD`p+5iA}<Nhey;jM`}I5*Fs*UrNTfCdHDz^TSwn`UE(RtTVEogcjzR~=L5g8>>gP4}cb(gCxjnlCv8K$nJb6x={m?R8uM+QJR#WkmpXh3K7uK7d#%*qHcZ<8VYsJ8hc?>nWR<GwkvpQAW(Jz6hw6GsIaK-a|;|JbQ>^$98q4c_=?2GQivH&Pn46I}eFlS}H2lkf+WcY!2u@n7vZs%nLocOR-FV-UgZ<CQJv0F0=m?htU-+9Og-Q?>a1>*-d2{AjW7;;eaJf%Z#wm+m37*+gShNRT8hRBMoh7fnNcgqDW59n++8>38|cAR1FY3mw5veErD%?)yaq)<+zrZy)VY$_!qPipk84M`ZT*(pnPIQmB3yug$_+W%^^k@#wDDDRgANVM&pF$tEnA(5fIGD*cG7T{8f$4hd!?MVT8s819g@g6-$h1P`WrJA%3x}`qd-i5>V;5QxcdNFVgE(_PC=iV^D5z7AgeJoP{j`D`t3(q>;(Ldgi)Uu1ATFAsY;qySG>O0Ja%y?}3?<$a*jermQqB8GRjwg@fp^G341jpq+#D*axY>L(1wrJ1~k)s>0^*I0~f>(fgW(c@Pc@5QKzk=&?=PrdKBd%!xW29#BNO*<+>g0Il+v#)akpIRE<E>i+5LzrNnphV*vzS-MY0IS=2`{R7hNr>mifrsUdgN2(OZ$*=V&|Km-oN4Neqoa8Xq{{W??OM(<3`z3}&u$$J|MwhAY!N@wiKqNI|%A+8uR(BwLt0>t3@-ixd3B624dg|%$ri$`>oA}ln;4wL_kp6~JoE5H6&p~V@QCvEJqP3OO3xGDLBLw&3IH_b6@wSxcN>7rac1(?SEl_+8jnpG4J8T!|SK_CD$&gct#IMq<hc!v<IN=*quBLdmc$^K2Dj}tX&XL%B3d(ahZE0%4thG!LsBx(mVsGJ8$H7dZsTx|blX}q@Y>h%}KG@k%Qbo~J2m&cce~s=9s;E+|*@vl2QDs)aKvt8~+8fj_>6z^T1_rGcc2ofjCe|o~4b-6})~JJ7<FzM9(F@y(Ize=<LTD9=A0e5ePQ-Hcsaw-4L3B``RcA%ZcAN}d$;t;P<Jcsboq!$yuoA=(z3jkmWE81X4<HdJb_;`^Mn;jLEJ~ugBtsu5L07&FLe#U>`PM3jN!nnK&{Rx<r#<Tiy?9ohZ&Py#F#rP$3XMFAX0l@QuQ5<o;J`AM<O7ky8gJ#|AYz5mvW72Kcq_jK^f;64vO;ND!7_`yl`~<^d2E-9Q(DeKTOZ0mSwnWq<dikgw@gl1#)B)omCKV;ma*p=Z)F9Rp2u6cIGA0dv@GJ>HMYwNZ{<vMyi93X0nRH7l#j&J=P^)LkoXE=WCf(3#|k+M+OH5s&P4qy)Q=U1zzX$a#aD1?e#e^g;UlRZD~^s8>c^#hAZx6UUxdVQo|k5g#PNX+oE7TF<$ODr=92tv{E)xPpTjatcrOpnJK=pl@8_L-wD*YPoh<m2wlGY1{}azJ`J22_OI(r-zABze!h5jZ&n2Ngur)3T&l=(VWjt$y^`b3sNm$?77S_lD6XaL$&TV0sZ1ChQ@J@LD-WE0r%R5={5c0ee-fzhBPBwWHSvJZRALAD0$`*el&s^a-Dhs|#p1JaFk0#4!S@3hRe3tinKQF*?+2B9D03(L?z%KA)IKS)#crq*_W(&)O_YL!`kqdf^e}+r)bF7g|M<bpaS9oUd;DsXuS7QCn3g}ylzxy@9<<{cVmLtyAqQK6J=Q=AgYc2fhBV(*SGLY&cL!!=$ds>NL`p7V)wOFKc0*=;VcFqgTxpef+nPD#*V_W_`p=9ERd|INJ*iAGOc@E7a{shfrOXv(A;<6C7@}WT|o5D-@kdY4!4%rk2!UufNaE-OV3^oK~Q&`28P>C%;5p19XAEvMnkgyPXzy=v?2@&9f0X7i)c`E-F(*6Q9eu3D&fx`ZOgO+{)0{RQpe2`DnGWp~zR{7uX#>ofL)K?J98u?@y+?>fvISUf4QdXAG(kf+T4M;7sS(Y)@#fdKGfwna^%gO^SXR%pUP~Zx~<QI8Q<>FxlYcHcLGfY;VKUoPzSiz}h#VM?z+w)i-E0}nN^>J~yeDQFJUju(%o?)^I=oiT+ABgo=DJvi71~@NJW$pQsWunWPUtyW(a`~50&We}0w4>tk0XQq3j}>0ZvQy+t%F3_rO1XSs(Xwx5g|f2pB*`xjj<m{4+2WGfz$#gIq2%4MLkkWdo>juSg%+41yt8Nnf8+wMnY=%KX^U%$<(e!wtXQte2fuN0fp{y{5w?lh^7Br!1-=Xa@(CZuwea!@%U0oEKH+1&wm9Q>4$Br7-4?dWf;*4pu&_?P3ou*w_fXy+jkfR}3d=y*;(lD<rEpHkkBxHM<O1axC=0KoY>1QF@=D74W9s-9Rkm<Vw!E|Qp;3ANA05Oj*W~?f<1M@t-o?zbRd~nq0*7UCN3XI~&U$hC%m=bp;sTe0|1KYOckw5oFCHZKi#)h|)~m`N`gHPHab{<}bi5M#wGzd(9IAC*_|#gA)OmqWD<MrQ!AUC-NdM8#$SwyA{RhPNtVHT;e>nB$-#GaI^2yJ<aq_VdI9uLgeSh@J!aJxNB2V}jlZ99iHooHmksMoMHnzO!`R<5|P0<eTe>szXBa@FxVB-(o|H9=a?my4+=ii*<Uzhx4UyS@EIPW~W{ok<Ae?VM58}#YnVSj!5wY~gg(Be;seSZ1Pu}{mk|L4Q+^z78y&yA64xSt&(MYx|IBSkn`%u8p)+E0~{YB<nB$HdWWXj2^D=B12WBi_(590P}xVbK)nGfUSYeP($(v<ye!p=CHG&q)~u=FvWLKOc9zTr4w)Ob`20bEF!MMMQec(urtO9Ixo49)@5<%Wz1S%nr*&qW$4$Nwf^dPon+d;5F?JM`ogJajYiN7W?ym$IHbsph#0J6^b^+aiUJ@VaQRm9*!kN%W!llT848~kZERp7AIvG{uLb;=is3AaAYl7569Y~<KpOBv>wi@Li@vUy=Yq;>5G(M-87^v)?Y)*a4a%1F4nn2>*2gSv<&C&IVr<X*XY<e4jb(=M`ojCxS!%YKBg@3jaI`k;7-agRJfBe3@45bopX!PDdhZPqzvmQqh&ac87;$+^+=yt_ZqE-WA@Q{IEo*cL)KqM+v50uv@OoLN9*B!uJw5Pu>3;WW$ve4kE>z*iKI8IW098OJdCsq$9p8R!@46$JuFF*44wVy+T)?Kgi6vK)&)wt!}&sK8IGYz+v0vwmvot9bJCG>Kfili4a@u_y<sV!CuNwQ^CeAjbWze{)?-X2kmZ)r?r`p9S`SA=rTyU;sk98|uO@A=^i@(1%VQ<uVhOFZ4Cf@L{o&ZJv>uKIOZ&|6VNc31#MqNE3`3Td;XM3wTpXR2*2De0^zl|@skdZ$SPm{N!;x`m8IF}py3EpZX+0cIm)64(c2CMMw<Tnn*=rM8hWn}R<7s9|!n8XaTbP#NXv1WBSpG1nhb0o9lwq!g$k18VF|CKAAJZ}%51EdOyNjaz;eHzZcq!O>D_RZ5eWu;vNYHdx92=U}!(Eoq{&3W3(iY31CT+1~>XUjHRy8fd(W_}0j%Q8#!`%jwv9qLXS`Ww0rv2fr7in7@m7CVXak@{+FywC9XYNju_J_ONJSoH6a?&y!qnz}Iy%Ht$uy>=h40lO-Qii!LrDZs(JMA;ac_(YZUcb_|ID$NFi(|^uwm8Z>sfXpxpO!f#(UZ2=yJIpgmT^z(;i&ht49CHzWw;A#+8^%b`lJkVg-yz^ciFTIcd1RwZ~+8lTx=u(EyINrkTqpv3r@<!gCz?*P~`A6L`&5ddm&GU#l<SnrnsO5IxH@Ff!4!?G0=Lrcm`Sz7uax8hG87gv2#%nv>q-T;-pMTN=PrTpb&^yUQN9bZ0H27hGRm^UH<Xu>RY-`U)y85KU}jNs()r9;D04PJ(ql;Ea;cZx&7_+?%UnXzWjx%+MlUB?(oI!<NoolUAnEijN-M`-EB+vyK#+|9-r>p?&)@acZ+^Kw1azLbnoHY&HnMX?{f6yp?qmy)9yREGRlpYK_2d&?n~Q!Ne^Gzp?~d9^z=5}{Qi%0eSOoQd~R-jd%REOAL!F><?idZhF;+F-yiPF?^DKZ6#IJj_<P-d-;c_781enSJpA{!F@h=!P>4tdz;U>7AzYz057*f#sbFkKN+|&ymn|~2tQs2u0w03WLqlLl@9fu6cgCF|aG)z_065PTHTJH?8A6|M;9{1(7l)%$>@aKqs_F=7Ft+)g4s6-c$f*0!&@SD4`;vagbno`JkL~&%%k<n5pz;W($U{%6>YY-6Bfx19x=ePQ7uX*#4134H8CPu17My+P%$8!$hsYJKrMuquWuI<#?R&a;ZfQUcScNSEhlHYGx&<pB85<<K9I-f%Yj)i8Dd0Pp0Tp5nO3H?i3rrNZ(9%P>yKnb9?BRJ#L-a$PTYW%7zGBzIuoz)7SwcaC+<>P_@4j?c-ez0_<G8k-g|WjpIqbxl9rS!29@G7GJEHGAEWBnTmR_@2;1em9Ph^(pL=l~9H~Z^-zG*MS*O#LyqaC}j+&qngcfyIap@`mUg)$jeOA07u_L4b=@k8GOhgK{>6mzs+wi&3}Nm<9`mayZ_^3_ruJI7_K6%2i?y3>PUbqir~X!Ue*7N&-s7;ah?CgqPGKmO}~0Ry=H+W', 'ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769', 35999), 'provenance': ('c-rlq>uw#lwa34orx5hnX?VZnRSpV)S6ipDWo*gD35vXXf0R9=w9RIk@vf?*{1U*TW`;wq^}oytm;QR4PLH=emB;HJuKVrd&BxRI=k_6eDDCF6ynb_?K2?7D_H_U0<IQb-xc=dPuipywoscT|Sab_LOJ9;S((~I-Lt2IAn`k;H8>9D0t8K_C;hXG6^b(9LxjYX!DNJ`M7nhC5-scjxA>S2uYQ_pzLuk5H`E^Jp=W92zc_&;?#@4XiP<atU_d)xfbycc|P`?go5qOAGu9Q@5>OvUvb;wliQKgC5N1IKI(ri~*ew){+RvMpOZBmQvm2Kx`O)yc2)}5?OB!3<9Dwf6sjFhz}9kOWqP!v9Ep@WSs1SOo;VH+yd7t>g}nw`qMcCk$(*ix%XPE=K-H#(^#z7EA2TN2G^9%Dq6+L~=B+2C|zQ<#M*F($9{*P-k}mbrG`AVn#4&ZU1H>N-|iEW6T+l#L7RKE!G$A-!-cvXHhZ?XUm*=DMcG^w1t}PVMdO!{h14n}>HvSx0m!*>%=E<Wva}V?1ju9IKszPa$%<me>*Rg*LGm<#N<9_LN)<O4_kB{r+x*;9sxB_|FK}hx<>b(w=@+(%o%7rN`U*56@9rbEwhel%8I`1-(U%U>%(f%`&87PKE&5hH7J9ly;_-;(YF{w&$TJl+LCnR?Pb7+qN&Fd&zTJFJ2Q5((AqjFdGqTtvZl2B9vl%-<~C-<q|T<8>hQAwUupOST6#Oo%E@87Fi~1x1p#^?KSq?%9x82aS!|4VhKrF>6k-zhF#i+x{z5Wk*b%Zly7a@O;vM>xi(8eauqq9w;8G)Wh3XLkEtf%y;tFR$QRbQ6;e!<P-NX^$Zbpz#+j<Mkj}Yk`aWc|$XN>BPZJEJ?e;8Ep(&eGkkw_SWwvTtc}<8U)=tT+f^xkQ&ifD{q{N6NU7}a~E9^r}Htl7U>?V6g=5L##ixYyW<m5UL0*3W1F?rb}#Q+9Fn0Bd++n+TOUHeR;BoztNmbO<mVmzz_QFV$gJ1fPu@_=In__(4~?X(HH>_a-Rdd>@C?P>o**|Esl%AIr$NtX=LhOl1+0<{o~m748VzK|;0P{`fbsI!Qj>FeO^eic|@b*{Cffj7A&zfZoHWL-DlhwK4-v@$(IqPHPNcDr>hZD~&jex1FwqFB?VX{~_K`%o1U=zti=tX68$`+Ie0!GepTRM1eRYRW!TEa1>h@NwXcPW$Ao%>2s_ZjeC~0XXhM3>X7WbQe;QY;@daXdN_TYHH9&leU0oTO3-?x@FM=Pcc%1^gDOhNFroYvTZ3<!m#d;Xu)zxG6!xX4A@fAmaw)G*s0W}l#B3J32u8YXH56q2I~lL37EyYT?K(f8I>j0gzADv_HFVm`xb2Vu&Gm(C98d790&NY^8!?XkD5u(Kt!|~?V_%rit0)AYPTDDA5v!a!9bQiMAwzwha!j@0N)PEgMLK%&Qmgm)?h9oj#~?MZzCfoNEPrx^}(=B<b6{DKu*9|Ye7s1L|V12MhJwSVc%<x0wIuu-Vh)p1D5$f{(}HPYZ!bRDuD?Q6n0QMG@JbvqM=GvHDIT5lDu#caa*V;gBGI0lhImfMUwmEBP=ar@6|k!94{BPX+X$6I`I^{=p0K}?pGz-h$v`__z|||te%Iex4jP31&Ip3wr^!j&9RCOHgaGk@K?6gh*`Hj@FWr>!f0SZ?}W}vYWBDD&I-zfIyORAuyPC@RM6-Knft!HNt0{dWGjY->7YQ|SdT!Do@C%+J?03r?h3&K8C&DET#7n?JiA^>rjub#7WR}3(xS`#D$mbj%0du<gaB``O+G~^(W;P|7D_tnfb~ssC@Bi-I8=zX4R-W-AMyogBL+cr?TQ!q`Klr<f(Qoku4(PS(`|@H)Bxv_kfCKF(Tja3vK>!s&ApTg&Tk8qttFC>C$OD{Yg3S00OX+_E2&p_s?)r7*mgMwbgN?|Hme4GpsBW@<TAXW5j6%JGF8=m8l{UQ6xj?u8HqyJH9!$Iry(j*0Eq&8<I+CVa*Q#A;Eaw8y0%S&+}0U-2Q`uhqM5~hmG;)6LD4T<TvH}(wjn>X?_e9&ocvSZ={CgslRq{A8v^Jf#ne7jigCqRDBVM+aP0=>g%vlF7Y%|fV@6tR%L^O88A7(Ms;h}O@%uDD?;4P}geeMJh-dyPKrSe5h8=*QiK@OWFS;CF?rs2pbP9kKVIPX9Nh!GziHyQ5e!ogvpaj>Utwry#nfq#h8Nf|*jI<|hW6gd;>uQfmcSQo0n16<B)w3NI*?`uL#G0;le^ydO_`<dkXQ*@3=a#UI^wDght67J^I=yXcI4GgR9{kMcI*6idH*A3KSXmOq5v?iNvZVuH$Um}+GJ|{Ie@ffR6Erm!)&un_P|&ip?Y&wf!5S$h2avjq5PTjg=yS5fG(y4qO7`tT4d_xr6>UA5L&)8#bm65Un2KQM6T!izZ$lDUiy%Y6INKpa?YCEkc)CXG6Z;Bv)Zi?+50Q6?+f{2anI~~N+YoSqO6rkV1>S76JEWkYk)deL2E3V;ZMXXjAusR(EbqBzs8-f(q54Qm%#g_#G#xZPXWQl85QL#ClVelZ1Y1QK1_vkn*iOq~EQhfi#&Q_TVJwHS9L91O%V8{su^h&77?*GukMB<H;obdRT_$6hjAb&G$yg?1nT%yJmdRKqW0{O)GM34>h{-sm+Yh%N-Y$=^JjU`E%VR8$u{_4|7|UZUkFh+)@)*lwe7nc^X`I(M<=a}@hnxPNKSV@6!*vGiPG~HmJOBFmB2m#wQpy}z8I&bkZFi?KjtcSS*c)z_%`-4dW&j6CsM#SBkT`J*QX;gI>9~#<jUwIZzV<ST9Dq;+8%ivert`NQb5fyqmeDnbnK9Nm*`8gI;Xx>H<Dddz0Q9sEh3>q>%MnafkruleRjo3QjKB(!#Yn!FeHy*l1_z-~(8LvK5&W;8w!+`Ym@x^+JuDi47<Yf80*c3G5i+?Fh6CBcO^=mFZ;(WTc08(ee-{5p9k)boMWMZi>~<65pPY>%^CZqG;4F5hf}OC4lURrTk|Ue8AyNz%S70^kcvSWKkt#4qkwn@wXTVqX$NOJDMTS6e(AKcoDS7m}mfNOS+nvZDY4?rw!qJ3hZH@`-P<E@}a<JL%Udke+@NS-A1*Ld~zU{;sco$Pg6Akf#ym4;dl-Xf6?Yv`dz!)|K36GxHhR9MpF0#mv*BT!M{I?x}Gq8<mnkjVd28~IzpN_*?5}cw2HxR2mFYn~w9FI0IHnbrEw1M&8ITj2|DBj9{Y#bUu+AVsc?(j*>Sr8$T-x0gY#wDY65r|<3_ng_4f8mf`>}05e@bM@D-;ci8Rz3}|%OshG^b39fq}Ya_C$zFpNJVU97kW1_#2ADKZiwbwhuO;RiY5mF0W?>%aBLSm?>drT*Ol1V9LG%9D}&b-J0(tz5%T(-E78m&<cnwFpJkxNtRj{gYY|1l@$Vu_C?~PQ7!TE;(_HqLFIvJ=0y7KBL+U(lS5n4-u7#%r$Vge^(=L~=rq~2@A?un#7k2+hz`BHS5D{<S05KjKx5;C9I_cN!Q14wOjLEKUqQg`SU=EU}xs-N*t;7r~q;zpgDq)Bd{j6?WYQ2x|Lqvb~B0ZnC90w`OcqunHv4YI|5HTk4TBtyH65iZZ_Ux3VR}RC(eF0HJ{Pgy$0to3Ev<yU)$a-M(Hk8I0OphA&1c@>>hP<l)?Ob9>25BOD99n!o5lTwc=7T3AOC6;5Jj4<S$KC`XJA4$IpPeK9I#itq0n!+fCy>>RB>MB4>v!$0-rRrsB|$ME`PaeZuZ5HUsKon%>^kGgLu7c6)?SjkUXq}WvCyKLe=F6zem?m{(!h(#Z*LheVV7BnY(hNei(N12E+>Vxi__Jl<~C!pNn!1Eklev?+@$6<Va>CJwS$CutQ>don%f~>f6)@zOhcTnp3UUN`Rduk*Pkz}UA}sDUggXRYZC!<tgv=**)*%UomEt`a@?e_c2I|%uenVm+N4DGlIrbPiEN_zCKauT96VOGIwT?|6|IB%a#EX`NY6=aYNARnU4WX&+?T9PO(gZCHg##WJu6$ik}A_NKQOB@y`WW?)TS=yA}-YzVA+UeBbJR=He%U`Wh0i2ST<tWh-D*|jaW9~w;%prPGUKU<s_DqSWaR&iRC1glUPn-If>;YmXr9+C;XR@SVm$QiDe{~kyu7z8Hr^imXTOSVi}2LBwpb${>fSWOZOt1oVuSh9Ge`-pY<X;*l_G@SFzct`$G-KCauE`^#_}^^E%pn>+GrfOE>14b;O#Sx<B9cYW62&Cta$J{dL*%U8;`#;n`Vlt3zF?X051RvftGCrc$%NwR`b?Q<Jm$FWFS;Sm&roUw}z_sF&;kHEZ;Aq?^;MA=0s~N0)AGbg-w;tkJ-so<e8-i1g%y|D{R(XzS2R)Ab*hlV6T_&$H`{LXWj<{Dack<+<zi6Vzg%-xmx0MyeSOR{e{I{}1(#`8PDde!+$`6aJf($j(#a!Rpx|LY)@Y&T;Luur{ON^EJ2g41Mu(+%d|Z)!Zf>ybsmfCK_T=2YV$Aw=dp6Xx2LWd>w4kAA8bvXrgisw&y#qjgHl(CW>lOo4UBzx_D!xSMpKf;&rg896VbhdqJf*Ev&tuh&<MkYSth7d^v8WO3#<$F5f!)P-m=5%iPPi+M4L*N%ic!q&`?!`yG4^x_Aq*^U8iwSetxb`j0mr`=`DD%V8{su^h&77|UTShp`;Sau~~DEQhfi#&Q_n;4uEVWxz5S%VaE*u}sD?8OvlWld(+3G8xNcER(TJ#<!S^f9xi(JjU`E%VR8$u{_4|7|UZUkFh+)@)*lwERXRmf1~jm8VyXEo4sTYg7e+dW-Zaqw<(+TAeb~7nDin$-*9Z!GT`D(z|MF2dM!P>Cfx*HvN6}B`_`nz*7-hKv-Sk%dk|c{`_`nz)}*Pyu@+mCRtU%X9n4y69qOYs>4-IHdo}4naI8z!`9=e?E>)9ORFlL1lP*=0ep8dqQIqyim+k@eqK%%;cXRqZG+#R3GGNkSY1Z}V7g`3)8=+3R2~1SOrJ4L7C;t}`@9&viUz8^gk>M-hu44pskcAf2{5Md|F+Ta_rgxapC$?zy=HcVrXdLj4&~n#5T$3DH4x}si^#+U!%R-T+LXU4$S~x+Xj(PyW&v1^K9Z0c;)Pr|p$3NcR-IgD@SNc?MAFtAfdNnxeJ2>j9-`~}%r$;|rrBl1wE%8-9-M_!u%5ZP4e%8L=&DEDfMsJ>Owg2ycSD5I>QL=k;eQI~<k#~8YKHT>9@ObksJ-qwU2BD_M^w1t}PVMdO!{h14o6qu3|MxI@B3vsuUZRIgXlZ4LTf)uR@tw#+J`izBm_<Ln9K>SSdLlaZ1k?s4zkHDRx`=wb*Ui)WD)F;NTwpaH%KfRGZa${d?c>j9FVL`2OBqyit-^W4ZzhktGOVZd-~#|p;xYH9YNl^6N9uYwp9XmRH17Sby-zov+v(w{O@!n}PaPzFwt(&bS^5v@?)$%_w{PznxqNr`9s5-N!rFgV?%#iGkGEr+zJEBC@3(=Q@oVqzAHT1+pKr&_Z{D}Z(`|Y9)5j4RRhBNBNJaw_RVWNdmo8$AR6-3H8<J8=02wF(>9wqy7}s1Cig$VB`Zvz&@1`?Ky~45CJiy|ViW=XFbZ=u$q*}~Kn&J%1YsUvpj7SYYv9ZlVw<8b7L&m+I9(0rLKE6xelYlq3A0FG=v2|abB-wpwZ62tls&@*>(eO4=mMFUiu$YIo9aZB;4cFO1)&pe(MbA%_nqPX-eebvBHr?H{&*|>Vlg4**h3Ly3(kck06rQ54HQvAhIv%bW^<!07b17<(2qd^Y<O06CGf(=vaQyP50g9lk)la_4SAYgk7g6;rflgQ#zAL@^0$T?&K2@pOTF=7R@j3STr=R}kZ?_;|t^', '6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f', 51113)}
TRIAL_BYTES = base64.b85decode(b'dm>?Dbai2DB075_V`yP=XJub?b7UeqZgp&IEFx%cb#ikeI&O7rY%C&kV{dY0B06q$Y-}tdb97;Jb#o#*Zgp&IEFyDsWpHyMIxu}KB4KiLX=Y(#bRs$;baH87Y%O*%A}k_fVQyq;WMOn=B075_VqtV+XkT+_dSxOyF)SivX?AIIX>V>{Wo}_&Y-MC3I%Z*Pb7d?dWpHm}XmcVuF)SixX>?y^a$#e1X>V>KIxsFVEFxxaY-A!jFf1Z$Wnpq|X>Mm<a$$63B04c;Eig4KB5h%KUvqS2aC0I$F)%PNEFyDbWo}_|X>TGrB6V(Qc4cyNVQeBSB6DSBWFk5-H!LD^aA9(DWnXS(XJK?{c4c38Woc(<bRs%1E-)}LEFyJoW^!d^dSxOyFnuf{V{dL_b#iiLZew{OIx#FFWMy_`Y;SOFWo~p|V_|S}B075_WMOG+d0%yNWFk5^E-)-2ZEtRLXl!|3b#r7QIyf*cFnuf{X>DzFbYWs_WnXD-aCLNZB075_VPs)&bY*g1b7)~QH8vtTB4Rl(V`DftV`DO8G&3+XWHDwpH8NpiWo9-qH8(OeF=Jw3H)Jz3Ff}w`IbkwnWMN@rW;JAGGdVD2HDxk1Ga@V^V`yb#YjAIAZggK`Z*FF3XJ2z@VKOx~B03^vIbt+rGdVV6H8?OhF=At3FgRgkW;il8HDY8iVq`I6GGsGmWHx1DFgY<{F*z|ZVKp%^He+HoWMwyEH85c!EFxoQWn*h_Z)t9HUvp?-GBq|LIwCVPH#spiWMw&HGc!0iGh<>hWo0%wI5RR~IAt_7H8Nv3IASqnV>UT5WH4r8W@R}wHZ(FaG&VP6Gh{R}Gc_VCB5-nVc4cm1ZewL%b7)~QH8vtTA~s<;F*7-0W->T2VmUZuGGjA1HaTNBW-&5mI51{nIW{#lW;iusVKHMcW;bRzF=jAfH8EjhWH)0oH)b(3W+E&ia%E>}b98cfUvP47YGq?|X>V>{b7)~QH8vtTB4uMaGGt;jWHn}FFgay7GG#b6H)3LDGcsc_WH4knF)%P>WHmKmIb&jFVK^~kHZn3cGh}3DFf%ndIAbv}H6kn`a%E>}b98cfUvp?-GBq|LIwCSLW;0?lW;kMDWH2)_G&DCfWHc}~HDqLAI5%N9Ib&fmWnngBV>dT2Gc-A4G%_|~IWsvqW?^AAIXE&gH8>(HB64MHWnpt=WM6b;ZE$aLVQgP@Ze?^|b7)~QH8vtTB4ju)Ib$<5WHd5nH83<WHZd?aG-PCDVPRo6VP$1CVl*%}Wi?`BVK-(uVl-tjWHn|vVlrlCGc+|eHa25nV<Idfb8u{FbYF95VKOx~B03^uWMVThGh<?9V`4EeWi~WnFg0d4Gd5;5Ff(E}V`4KfHZV0YG&5ypGh#DkHZ);kIWsjlWnnpFIALZuH#RvUEFyGrVQFq<a$j?3VKOx~B03^sH92K9G&VP6W;HivV=-oCVlXs0VmLT7G&VUiHDNe2H8Nu`Ib~#GIAJz4WH~rEGBz|YWHn_uHZo#4GcYzHeJmnyY+-a}Z*py4V_|S}B075_WMOG+d0%yNWFk5-FfK4GB5iMObZBgOUv+b2B04cJFfK5CEFy4nWn^h%bY)~;cW-iYbYEj(b7dkrdm>|KVRC0>Uv+b2B04ZGGb|!#Z*_8WB04ZGH#L1MB5-nPc42g7UvF+~c_KP=a&=`aB6DMCWo=<!c4cyNX>V>KIx#FFb7OCEWn^D-bz*I4b8~5LZX!BnVQh0{EFyDnb#h~6Uuko1Y+-b1Z*C$wdm>?VWNCC?b#rB8B06SaY;$ESB4%lHUvF+~c_KP=a&=`aB4}l7WM6M}bYFFIWn>~cW?^h|Wh^3TWMyu2X>@sCb8mHWV`U;bB4ur2Wn^S!WFT^NZggpFWgv28XK8bEa(N(da&Ky7V{~b6ZaW}mb!==Pa%E>}b98cfAX7*|EiyGWAYpD}Xm4^NEFyGeb97&Ib7f>AI%Z*Pb7g%jB6D<MbaitgIwEjkV{2h&WnXS@bYE;?b#7y5Wn?0K3I')
TRIAL_SHA256 = '01083d1e5f1f2f5bc62e6de8336499f2b00295042987884d656f3c41e100b5c2'

def embedded_manifest(name: str) -> dict[str, Any]:
    try: raw, expected, size = EMBEDDED[name]
    except KeyError as exc: raise LaunchError("unknown embedded manifest") from exc
    try: data = zlib.decompress(base64.b85decode(raw.encode("ascii")))
    except Exception as exc: raise LaunchError("embedded manifest decode failed") from exc
    if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
        raise LaunchError("embedded manifest hash mismatch")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict): raise LaunchError("embedded manifest is not an object")
    return value

PROJECTION_FIELDS = ("group_id", "video_count", "contains_nontrain_member", "video_ids", "splits")
def runtime_registry_projection() -> dict[str, Any]:
    projection = embedded_manifest("registry_projection")
    if (projection.get("artifact") != "runtime-registry-projection-v1"
            or projection.get("source_registry_sha256") != REGISTRY_SHA256):
        raise LaunchError("runtime registry projection provenance mismatch")
    groups = projection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise LaunchError("runtime registry projection groups are invalid")
    normalized = []
    for item in groups:
        if not isinstance(item, Mapping) or set(item) != set(PROJECTION_FIELDS):
            raise LaunchError("runtime registry projection fields are invalid")
        if (not isinstance(item["group_id"], str) or not isinstance(item["video_count"], int)
                or not isinstance(item["contains_nontrain_member"], bool)
                or not isinstance(item["video_ids"], list) or not all(isinstance(value, str) for value in item["video_ids"])
                or not isinstance(item["splits"], list) or not all(isinstance(value, str) for value in item["splits"])):
            raise LaunchError("runtime registry projection values are invalid")
        normalized.append(dict(item))
    canonical = {"artifact": "runtime-registry-projection-v1", "source_registry_sha256": REGISTRY_SHA256, "groups": normalized}
    if _canonical_json(projection) != _canonical_json(canonical):
        raise LaunchError("runtime registry projection is not canonical")
    if hashlib.sha256(_canonical_json(canonical)).hexdigest() != REGISTRY_PROJECTION_SHA256:
        raise LaunchError("runtime registry projection hash mismatch")
    return canonical


def trial_record() -> dict[str, Any]:
    if hashlib.sha256(TRIAL_BYTES).hexdigest() != TRIAL_SHA256: raise LaunchError("trial record hash mismatch")
    value = json.loads(TRIAL_BYTES.decode("ascii"))
    if _canon(value) != TRIAL_BYTES or value.get("status") != "package_not_launched":
        raise LaunchError("trial record is not canonical and immutable")
    return value

# Released TemporalUNet3D implementation
"""Simple 3D+T U-Net.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z, Y, X)``

"""


import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class _TemporalAttention(nn.Module):
    """Per-voxel multi-head self-attention across time."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Z, Y, X)
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)

        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    """Minimal 3D temporal U-Net.

    Parameters
    ----------
    in_channels : int
        Input channels per frame.
    out_channels : int
        Output feature channels per frame.
    layers : sequence of int
        Encoder channel widths, shallow to deep. Number of stages equals
        ``len(layers)``; spatial size is halved before every stage except
        the first.
    gradient_checkpointing : bool
        If True (default), wrap encoder/decoder conv blocks with
        ``torch.utils.checkpoint`` during training to reduce activation
        memory at the cost of recomputing activations in the backward
        pass.
    skip_fullres_temporal : bool
        If True (default), replace the temporal-attention block at the
        full-resolution (first) encoder stage with an Identity. Per-voxel
        attention at full res dominates both memory and runtime; skipping
        it gives ~3x speedup and ~30% less memory with negligible quality
        loss in practice.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing: bool = True,
        skip_fullres_temporal: bool = True,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")

        self.gradient_checkpointing = gradient_checkpointing

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(_conv_block(prev, ch))
            if skip_fullres_temporal and i == 0:
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(ch))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.decoder_blocks.append(_conv_block(layers[i] + layers[i - 1], layers[i - 1]))

        self.head = nn.Conv3d(layers[0], out_channels, kernel_size=1)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z, Y, X)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])

        skips: list[torch.Tensor] = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0:
                x = self.pool(x)
            x = self._run(block, x)
            x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._run(block, x)

        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])

# Adapter behavior
"""Released-core temporal adapter with device-correct sparse supervision."""


from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_SHA256 = "347915de9c33883cb2ee69832a8e4552c88b1ec692d0fbfe956422467d3d4235"
WEIGHTS_BYTES = 8_357_783
CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
CONFIG_BYTES = 165
STATE_SCHEMA_SHA256 = "a832aed27d60ec876a2be8417bbef67718959d42c3c2e57a59a4a67e6a150d8b"
EXPECTED_CONFIG = {"unet_out_channels": 32, "unet_layers": [32, 64, 128], "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 5.0}
SCENARIOS = ("universal", "sparse_clean", "dense_crowded", "low_snr_depth")
OUTPUT_SCHEMA = ("center_logit", "backward_dz_um", "backward_dy_um", "backward_dx_um", "log_variance", "division_logit")
CENTER_SUPERVISION = "annotated-only"
CENTER_LOSS = "report-only-excluded"


class CheckpointError(ValueError):
    """Raised when the attested released checkpoint is not exact."""


class TargetError(ValueError):
    """Raised when sparse labels cannot be represented unambiguously."""


@dataclass(frozen=True)
class CoreLoad:
    tensor_count: int
    tensor_numel: int
    schema_sha256: str
    discarded_tensor_count: int
    discarded_numel: int


@dataclass(frozen=True)
class SparseTargets:
    """Voxel targets allocated on the prediction device, never CPU by default."""

    center: torch.Tensor
    center_supervision_mask: torch.Tensor
    displacement: torch.Tensor
    displacement_mask: torch.Tensor
    division: torch.Tensor
    division_mask: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.center.device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise CheckpointError(f"missing required file: {path}")
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha256:
        raise CheckpointError(f"identity mismatch for {path}")


def verify_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    _verify_file(path, CONFIG_BYTES, CONFIG_SHA256)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("released config is not valid JSON") from exc
    if config != EXPECTED_CONFIG:
        raise CheckpointError("released config content mismatch")
    return config


def verify_released_files(weight_path: str | Path, config_path: str | Path) -> None:
    verify_config(config_path)
    _verify_file(Path(weight_path), WEIGHTS_BYTES, WEIGHTS_SHA256)


def _finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _load_state(weight_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(weight_path)
    _verify_file(path, WEIGHTS_BYTES, WEIGHTS_SHA256)
    try:
        first = torch.load(path, map_location="cpu", weights_only=True)
        second = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError("weights_only CPU loading failed") from exc
    if not isinstance(first, dict) or not isinstance(second, dict) or tuple(first) != tuple(second) or len(first) != 136:
        raise CheckpointError("checkpoint tensor schema drift")
    for name, tensor in first.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not _finite(tensor):
            raise CheckpointError("checkpoint contains an invalid tensor")
        if not torch.equal(tensor, second[name]):
            raise CheckpointError("checkpoint double-load was not identical")
    return first


def _partition_state(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {"unet": {}, "detect_head": {}, "transformer": {}}
    for name, value in state.items():
        prefix, sep, suffix = name.partition(".")
        if not sep or prefix not in groups or not suffix:
            raise CheckpointError(f"unknown checkpoint key: {name}")
        groups[prefix][suffix] = value
    if not all(groups.values()):
        raise CheckpointError("checkpoint lacks a released component")
    return groups


def load_released_core(backbone: nn.Module, weight_path: str | Path, config_path: str | Path) -> tuple[nn.Conv3d, CoreLoad]:
    """Strictly restore released U-Net and detector, discarding association weights."""
    verify_config(config_path)
    state = _load_state(weight_path)
    groups = _partition_state(state)
    if len(state) != 136 or sum(v.numel() for v in state.values()) != 2_077_996:
        raise CheckpointError("checkpoint tensor accounting mismatch")
    if sum(v.numel() for v in groups["transformer"].values()) != 580_353:
        raise CheckpointError("discarded component accounting mismatch")
    try:
        backbone.load_state_dict(groups["unet"], strict=True)
        detector = nn.Conv3d(32, 1, kernel_size=1)
        detector.load_state_dict(groups["detect_head"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError("strict released-core restore failed") from exc
    return detector, CoreLoad(len(state), sum(v.numel() for v in state.values()), STATE_SCHEMA_SHA256,
                              len(groups["transformer"]), sum(v.numel() for v in groups["transformer"].values()))


class _Specialist(nn.Module):
    """Identity-start residual specialist; exits are exactly zero initialized."""

    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(32, 8, 1)
        self.expand = nn.Conv3d(8, 32, 1)
        self.calibrate = nn.Conv3d(6, 6, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.zeros_(self.calibrate.weight)
        nn.init.zeros_(self.calibrate.bias)

    def forward(self, features: torch.Tensor, universal: torch.Tensor, aux_head: nn.Module) -> torch.Tensor:
        adjusted = features + self.expand(F.gelu(self.reduce(features)))
        branch = torch.cat((universal[:, :1], aux_head(adjusted)), dim=1)
        return branch + self.calibrate(branch)


class TemporalAdapterModel(nn.Module):
    """Released temporal features plus a universal head and three specialists."""

    def __init__(self, backbone: nn.Module, detect_head: nn.Conv3d) -> None:
        super().__init__()
        self.backbone = backbone
        self.detect_head = detect_head
        self.aux_head = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, 5, 1))
        self.specialists = nn.ModuleDict({"sparse_clean": _Specialist(), "dense_crowded": _Specialist(), "low_snr_depth": _Specialist()})
        self.set_unfreeze_depth(0)

    @staticmethod
    def scenarios() -> tuple[str, ...]:
        return SCENARIOS

    def set_unfreeze_depth(self, depth: int) -> dict[str, int]:
        if depth not in (0, 1, 2):
            raise ValueError("unfreeze depth must be 0, 1, or 2")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.detect_head.parameters():
            parameter.requires_grad = False
        if depth >= 1:
            if not hasattr(self.backbone, "head"):
                raise ValueError("released backbone must expose head")
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True
        if depth >= 2:
            blocks = getattr(self.backbone, "decoder_blocks", None)
            if not isinstance(blocks, nn.ModuleList) or not blocks:
                raise ValueError("released backbone must expose decoder blocks")
            for parameter in blocks[-1].parameters():
                parameter.requires_grad = True
        self.detect_head.eval()
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self.parameter_accounting()

    def parameter_accounting(self) -> dict[str, int]:
        parameters = tuple(self.parameters())
        total = sum(p.numel() for p in parameters)
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable,
                "backbone_trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                "detect_head_trainable": sum(p.numel() for p in self.detect_head.parameters() if p.requires_grad),
                "aux_head_trainable": sum(p.numel() for p in self.aux_head.parameters() if p.requires_grad)}

    def train(self, mode: bool = True) -> "TemporalAdapterModel":
        super().train(mode)
        self.detect_head.eval()
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, scenario: str = "universal") -> torch.Tensor:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[2] != 1 or min(images.shape) < 1:
            raise ValueError("images must have nonempty shape (B,T,1,Z,Y,X)")
        if not _finite(images):
            raise ValueError("images must be finite")
        batch, steps, _, z, y, x = images.shape
        features = self.backbone(images)
        if tuple(features.shape) != (batch, steps, 32, z, y, x):
            raise ValueError("released backbone returned an incompatible feature shape")
        flat = features.reshape(batch * steps, 32, z, y, x)
        universal = torch.cat((self.detect_head(flat), self.aux_head(flat)), dim=1)
        output = universal if scenario == "universal" else self.specialists[scenario](flat, universal, self.aux_head)
        if tuple(output.shape) != (batch * steps, 6, z, y, x):
            raise RuntimeError("adapter output schema violation")
        return output.reshape(batch, steps, 6, z, y, x)


def _require_int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetError(f"{name} must be an integer")
    return value


def _require_float(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TargetError(f"{name} must be finite")
    return float(value)


def _voxel(row: Mapping[str, Any], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    value = tuple(int(round(_require_float(row, axis))) for axis in ("z", "y", "x"))
    if any(index < 0 or index >= limit for index, limit in zip(value, shape)):
        raise TargetError("node coordinate is outside the image")
    return value


def _target_spec(output_or_shape: torch.Tensor | Sequence[int], device: torch.device | str | None) -> tuple[tuple[int, ...], torch.device]:
    if isinstance(output_or_shape, torch.Tensor):
        if device is not None and torch.device(device) != output_or_shape.device:
            raise TargetError("explicit target device disagrees with prediction device")
        return tuple(int(v) for v in output_or_shape.shape), output_or_shape.device
    if device is None:
        raise TargetError("shape-only target construction requires an explicit device")
    return tuple(int(v) for v in output_or_shape), torch.device(device)


def build_sparse_targets(output_or_shape: torch.Tensor | Sequence[int], nodes: Iterable[Mapping[str, Any]],
                         edges: Iterable[Mapping[str, Any]], voxel_size_um: Sequence[float] = (1.625, 0.40625, 0.40625),
                         *, device: torch.device | str | None = None) -> SparseTargets:
    """Build sparse voxel labels on the output device; predictions never move."""
    output_shape, target_device = _target_spec(output_or_shape, device)
    if len(output_shape) != 6 or output_shape[2] != 6 or min(output_shape) < 1:
        raise TargetError("output must have positive shape (B,T,6,Z,Y,X)")
    batch_size, steps, _, z, y, x = output_shape
    scale = tuple(float(v) for v in voxel_size_um)
    if len(scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in scale):
        raise TargetError("voxel_size_um must be three positive finite values")
    target_shape = (batch_size, steps, 1, z, y, x)
    center = torch.zeros(target_shape, dtype=torch.float32, device=target_device)
    center_supervision_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    displacement = torch.zeros((batch_size, steps, 3, z, y, x), dtype=torch.float32, device=target_device)
    displacement_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    division = torch.zeros(target_shape, dtype=torch.float32, device=target_device)
    division_mask = torch.zeros(target_shape, dtype=torch.bool, device=target_device)
    records: dict[int, tuple[int, int, tuple[int, int, int]]] = {}
    occupied: set[tuple[int, int, tuple[int, int, int]]] = set()
    for row in nodes:
        if not isinstance(row, Mapping):
            raise TargetError("node rows must be mappings")
        node_id, batch, time = _require_int(row, "id"), _require_int(row, "batch"), _require_int(row, "time")
        if node_id in records or not (0 <= batch < batch_size and 0 <= time < steps):
            raise TargetError("duplicate node id or invalid node batch/time")
        point = _voxel(row, (z, y, x))
        key = (batch, time, point)
        if key in occupied:
            raise TargetError("two annotations collapse to one target voxel")
        occupied.add(key)
        records[node_id] = (batch, time, point)
        center[batch, time, 0, *point] = 1.0
        center_supervision_mask[batch, time, 0, *point] = True
    children: dict[int, list[int]] = {}
    parent_for_child: dict[int, int] = {}
    for row in edges:
        if not isinstance(row, Mapping):
            raise TargetError("edge rows must be mappings")
        source, target = _require_int(row, "source_id"), _require_int(row, "target_id")
        if source not in records or target not in records:
            raise TargetError("edge endpoint is missing from nodes")
        if target in parent_for_child:
            raise TargetError("child has more than one annotated parent")
        parent_for_child[target] = source
        children.setdefault(source, []).append(target)
    for child, parent in parent_for_child.items():
        pb, pt, pp = records[parent]
        cb, ct, cp = records[child]
        if pb != cb or ct != pt + 1:
            raise TargetError("edges must join consecutive frames in one batch")
        vector = tuple((a - b) * unit for a, b, unit in zip(pp, cp, scale))
        displacement[cb, ct, :, *cp] = torch.tensor(vector, dtype=torch.float32, device=target_device)
        displacement_mask[cb, ct, 0, *cp] = True
    for parent, child_ids in children.items():
        batch, time, point = records[parent]
        division[batch, time, 0, *point] = float(len(child_ids) > 1)
        division_mask[batch, time, 0, *point] = True
    return SparseTargets(center, center_supervision_mask, displacement, displacement_mask, division, division_mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum() / mask.sum().to(values.dtype).clamp_min(1.0)


def multitask_loss(output: torch.Tensor, targets: SparseTargets, division_enabled: bool = False) -> dict[str, torch.Tensor]:
    """Report center loss on annotations only; optimize sparse auxiliary heads."""
    if output.device != targets.device:
        raise TargetError("prediction and sparse targets must share a device")
    if tuple(output.shape) != tuple(targets.center.shape[:2]) + (6,) + tuple(targets.center.shape[3:]):
        raise TargetError("output and target shapes differ")
    if targets.center_supervision_mask.dtype is not torch.bool:
        raise TargetError("center supervision mask must be boolean")
    if targets.center_supervision_mask.device != output.device:
        raise TargetError("center supervision mask must share the prediction device")
    if tuple(targets.center_supervision_mask.shape) != tuple(targets.center.shape):
        raise TargetError("center supervision mask shape differs from center target")
    center_terms = F.binary_cross_entropy_with_logits(output[:, :, :1], targets.center, reduction="none")
    center = _masked_mean(center_terms, targets.center_supervision_mask)
    error = (output[:, :, 1:4] - targets.displacement).abs().sum(dim=2, keepdim=True)
    displacement = _masked_mean(error, targets.displacement_mask)
    logged = output[:, :, 4:5].clamp(-10.0, 10.0)
    variance = _masked_mean(0.5 * (torch.exp(-logged) * error + logged), targets.displacement_mask)
    division = (_masked_mean(F.binary_cross_entropy_with_logits(output[:, :, 5:6], targets.division, reduction="none"), targets.division_mask)
                if division_enabled else output.new_zeros(()))
    return {"total": displacement + variance + division, "center": center,
            "displacement": displacement, "log_variance": variance, "division": division}

# Bounded trainer behavior
SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
PROVENANCE_SHA256 = "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f"
REGISTRY_SHA256 = "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158"
REGISTRY_PROJECTION_SHA256 = "ec92db4d5fd09e82e867bbf32c1d0d8100dd55a9cbea81c62263ddf03598c115"
LRS = (1e-5, 3e-5, 1e-4)
UNFREEZE_DEPTHS = (0, 1, 2)
FRACTIONS = (0.1, 0.3, 1.0)
FIRST_CANDIDATE = {"scenario": "universal", "unfreeze": 0, "lr": 1e-5, "fraction": 0.1}
MAX_HOURS_CAP, MAX_CHARGE_CAP, MAX_EPOCHS, MAX_STEPS = 0.75, 0.30, 10, 1_000
CHECKPOINT_EVERY_STEPS = 16
BATCH_SCHEMA = ("frames", "nodes", "edges", "voxel_size_um", "window_id")
USE_MIXED_PRECISION = False


class LaunchError(ValueError):
    """Raised before a disallowed source, incompatible resume, or unsafe step."""


_STEM = re.compile(r"^[0-9a-f]{4}_[0-9a-f]{8}$")


@dataclass(frozen=True)
class SourceRecord:
    """Immutable fit-only registry resolution, never a filesystem-derived ID."""

    group_id: str
    video_id: str
    stem: str

    def json(self) -> dict[str, str]:
        return {"group_id": self.group_id, "video_id": self.video_id, "stem": self.stem}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _atomic_checkpoint(path: Path, state: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _sha256_file(path)


def _load_json_exact(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file() or _sha256_file(file_path) != expected_sha256:
        raise LaunchError(f"immutable manifest mismatch: {file_path}")
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid manifest: {file_path}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"manifest is not an object: {file_path}")
    return value


def _fold_data(fold: int, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> tuple[tuple[str, ...], set[str]]:
    if fold not in (0, 1, 2):
        raise LaunchError("fold must be one of 0, 1, 2")
    try:
        split_fold, provenance_fold = split["folds"][str(fold)], provenance["folds"][str(fold)]
        fit, held = set(split_fold["fit_group_ids"]), set(split_fold["held_out_group_ids"])
        audit, forbidden = set(split["audit"]["group_ids"]), set(provenance_fold["forbidden_fit_group_ids"])
    except (KeyError, TypeError) as exc:
        raise LaunchError("split/provenance schema is incomplete") from exc
    if not fit or fit & held or fit & audit or fit & forbidden or not held <= forbidden or not audit <= forbidden:
        raise LaunchError("fit-source isolation invariant failed")
    return tuple(sorted(fit)), held | audit | forbidden


def select_fit_groups(fold: int, split: Mapping[str, Any], provenance: Mapping[str, Any], fraction: float, seed: int,
                      requested_groups: Iterable[str] | None = None) -> tuple[str, ...]:
    if fraction not in FRACTIONS:
        raise LaunchError("fraction is outside the fixed search grid")
    fit, forbidden = _fold_data(fold, split, provenance)
    fit_set, requested = set(fit), tuple(requested_groups or ())
    for group in requested:
        if group not in fit_set:
            raise LaunchError(f"{'forbidden' if group in forbidden else 'unknown'} group access rejected: {group}")
    candidates = tuple(sorted(set(requested))) if requested else fit
    ranked = sorted(candidates, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("ascii")).hexdigest())
    selected = tuple(ranked[:max(1, math.ceil(len(ranked) * fraction))])
    if not selected or set(selected) - fit_set or set(selected) & forbidden:
        raise LaunchError("selected sources are not fit-only")
    return selected


def resolve_fit_records(selected: Sequence[str], registry: Mapping[str, Any], forbidden: set[str]) -> tuple[SourceRecord, ...]:
    """Map attested split group IDs to one unique train stem each.

    The registry, rather than a group-ID-shaped pathname, is the only bridge
    between split identities and competition source layout.
    """
    groups = registry.get("groups")
    if not isinstance(groups, list):
        raise LaunchError("registry groups is not a list")
    selected_ids = tuple(selected)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise LaunchError("selected registry groups are missing or duplicate")
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for item in groups:
        if not isinstance(item, Mapping) or not isinstance(item.get("group_id"), str):
            raise LaunchError("registry group record is malformed")
        indexed.setdefault(item["group_id"], []).append(item)
    resolved: list[SourceRecord] = []
    stems: set[str] = set()
    for group_id in selected_ids:
        if group_id in forbidden:
            raise LaunchError(f"forbidden registry group: {group_id}")
        matches = indexed.get(group_id, [])
        if len(matches) != 1:
            raise LaunchError(f"registry mapping missing or duplicate: {group_id}")
        item = matches[0]
        video_ids, splits = item.get("video_ids"), item.get("splits")
        if (item.get("video_count") != 1 or item.get("contains_nontrain_member") is not False
                or not isinstance(video_ids, list) or len(video_ids) != 1
                or not isinstance(splits, list) or splits != ["train"]):
            raise LaunchError(f"registry mapping is not one train video: {group_id}")
        video_id = video_ids[0]
        if not isinstance(video_id, str) or not video_id.startswith("train:") or video_id.startswith("test:"):
            raise LaunchError(f"registry video is non-train or malformed: {group_id}")
        stem = video_id.removeprefix("train:")
        if not _STEM.fullmatch(stem) or stem in stems:
            raise LaunchError(f"registry stem is malformed or duplicate: {group_id}")
        stems.add(stem)
        resolved.append(SourceRecord(group_id=group_id, video_id=video_id, stem=stem))
    return tuple(resolved)


def _validate_budget(maximum_hours: float, maximum_charge: float) -> None:
    if not (0 < maximum_hours <= MAX_HOURS_CAP) or not (0 <= maximum_charge <= MAX_CHARGE_CAP):
        raise LaunchError("requested time or charge exceeds the prelaunch cap")


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    return {"fold": args.fold, "scenario": args.scenario, "unfreeze": args.unfreeze, "lr": args.lr,
            "fraction": args.fraction, "seed": args.seed,
            "center_supervision": CENTER_SUPERVISION, "center_loss": CENTER_LOSS,
            "split_sha256": SPLIT_SHA256, "provenance_sha256": PROVENANCE_SHA256, "registry_sha256": REGISTRY_SHA256, "registry_projection_sha256": REGISTRY_PROJECTION_SHA256,
            "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256}


def _device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def prelaunch_manifest(args: argparse.Namespace, selected: Sequence[SourceRecord], device: str) -> dict[str, Any]:
    identity = {**_identity(args), "selected_sources": [record.json() for record in selected]}
    return {"artifact": "prelaunch-v1", "status": "not_launched" if args.dry_run else "armed",
            "identity": identity, "source_isolation": {"fit_only": True,
            "selected_group_ids": [record.group_id for record in selected],
            "selected_train_video_ids": [record.video_id for record in selected],
            "selected_sources": [record.json() for record in selected],
            "audit_used": False, "held_out_used": False, "test_used": False},
            "execution": {"device": device, "mixed_precision": device == "cuda", "gradient_clip_norm": 1.0,
            "epochs": args.epochs, "max_steps": args.max_steps, "batch_size": 1, "dry_run": args.dry_run},
            "spend_gate": {"maximum_hours": args.maximum_hours, "maximum_charge": args.maximum_charge,
            "cap_hours": MAX_HOURS_CAP, "cap_charge": MAX_CHARGE_CAP, "approved_spend": False},
            "retained_path": {"scenario": "universal", "edge_only": True, "division_enabled": False,
            "specialists": "identity_until_t05"}}


def _row_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaunchError(f"GEFF {key} must be an integer")
    return value


def _row_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LaunchError(f"GEFF {key} must be finite")
    return float(value)


def _quantile(quantiles: Mapping[str, Any], value: float) -> float:
    for key, raw in quantiles.items():
        try:
            if abs(float(key) - value) < 1e-9 and math.isfinite(float(raw)):
                return float(raw)
        except (TypeError, ValueError):
            continue
    raise LaunchError("Zarr quantile is missing or nonfinite")


ZARR_ROOTS = (
    Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1'),
    Path('/kaggle/input/biohub-tracking-support-pack-50ep-v1'),
)
ZARR_DEPS = Path('/kaggle/working/t03/deps-v1')
ZARR_WHEELS = (
    ('zarr-3.2.1-py3-none-any.whl', 319617, 'f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7'),
    ('donfig-0.8.1.post1-py3-none-any.whl', 21592, '2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d'),
    ('google_crc32c-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl', 33364, '14f87e04d613dfa218d6135e81b78272c3b904e2a7053b841481b38a7d901411'),
    ('numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 8879163, 'c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3'),
    ('deprecated-1.3.1-py2.py3-none-any.whl', 11298, '597bfef186b6f60181535a29fbe44865ce137a5079f295b479886c82729d5f3f'),
    ('wrapt-2.2.2-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl', 169828, '2e096c9d39a59b35b63c9aacfbbbec2088ff51ff1fc31051acc60a07f42f273a'),
    ('typing_extensions-4.15.0-py3-none-any.whl', 44614, 'f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548'),
    ('packaging-26.2-py3-none-any.whl', 100195, '5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e'),
    ('pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl', 807870, 'ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc'),
)
RUNTIME_IMPORTS = ('donfig','google_crc32c','numcodecs','deprecated','wrapt','typing_extensions','packaging','yaml','zarr')
RUNTIME_NATIVE_IMPORTS = ('google_crc32c._crc32c','numcodecs.blosc','yaml._yaml','wrapt._wrappers')
RUNTIME_DISTS = {
    'zarr':'3.2.1','donfig':'0.8.1.post1','google-crc32c':'1.8.0','numcodecs':'0.15.1',
    'Deprecated':'1.3.1','wrapt':'2.2.2','typing-extensions':'4.15.0','packaging':'26.2','PyYAML':'6.0.3',
}
_RUNTIME_ZARR: Any=None

def _wheel_root(specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS,
                roots: Sequence[Path]=ZARR_ROOTS) -> Path:
    hits = []
    for base in roots:
        directory = Path(base) / 'wheels'
        if not directory.is_dir(): continue
        if all((directory / name).is_file() for name, _, _ in specs):
            resolved = directory.resolve(strict=True)
            if resolved not in hits: hits.append(resolved)
    if len(hits) != 1: raise LaunchError('support wheel search must yield one exact directory')
    root = hits[0]
    for name, size, digest in specs:
        path = root / name
        if path.stat().st_size != size or _sha_file(path) != digest:
            raise LaunchError('support wheel identity mismatch')
    return root

def _extract_wheels(root: Path, dest: Path,
                    specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS) -> None:
    identity = json.dumps({'schema_version':1,'wheels':[list(item) for item in specs]},sort_keys=True,separators=(',',':')) + '\n'
    marker = hashlib.sha256(identity.encode('ascii')).hexdigest() + '\n'
    ready = dest / '.ready'
    if ready.is_file() and ready.read_text('ascii') == marker: return
    temporary = dest.with_name(dest.name + '.tmp')
    if temporary.exists(): shutil.rmtree(temporary)
    if dest.exists(): shutil.rmtree(dest)
    temporary.mkdir(parents=True)
    try:
        for name, _, _ in specs:
            with zipfile.ZipFile(root / name) as archive:
                for info in archive.infolist():
                    parts = Path(info.filename).parts
                    mode = (info.external_attr >> 16) & 0o170000
                    if (not parts or info.filename.startswith(('/', '\\')) or '\\' in info.filename
                            or any(part in ('', '.', '..') for part in parts) or mode == 0o120000):
                        raise LaunchError('unsafe support wheel member')
                    target = temporary.joinpath(*parts)
                    try: target.resolve().relative_to(temporary.resolve())
                    except ValueError as exc: raise LaunchError('support wheel member escapes extraction root') from exc
                    if info.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open('wb') as output: shutil.copyfileobj(source, output)
        (temporary / '.ready').write_text(marker, encoding='ascii')
        temporary.replace(dest)
    except Exception as exc:
        if temporary.exists(): shutil.rmtree(temporary)
        if isinstance(exc, LaunchError): raise
        raise LaunchError('offline support wheel extraction failed') from exc

def _ensure_zarr(specs: Sequence[tuple[str, int, str]]=ZARR_WHEELS,
                 roots: Sequence[Path]=ZARR_ROOTS, dest: Path=ZARR_DEPS,
                 importer: Any=None) -> Any:
    global _RUNTIME_ZARR
    if importer is None and _RUNTIME_ZARR is not None: return _RUNTIME_ZARR
    load = importlib.import_module if importer is None else importer
    root = _wheel_root(specs, roots); _extract_wheels(root, Path(dest), specs)
    location = str(Path(dest))
    if location in sys.path: sys.path.remove(location)
    sys.path.insert(0, location)
    try:
        modules = {name:load(name) for name in RUNTIME_IMPORTS}
        if importer is None:
            for name in RUNTIME_NATIVE_IMPORTS: load(name)
            for name, expected in RUNTIME_DISTS.items():
                if importlib.metadata.version(name) != expected: raise LaunchError('offline dependency version mismatch')
            try: Path(modules['zarr'].__file__).resolve().relative_to(Path(dest).resolve())
            except (AttributeError, ValueError) as exc: raise LaunchError('zarr did not load from verified offline wheels') from exc
        if importer is None: _RUNTIME_ZARR=modules['zarr']
        return modules['zarr']
    except LaunchError: raise
    except (ImportError, OSError, RuntimeError) as exc:
        raise LaunchError('runtime dependency import failed after verified offline bootstrap') from exc


class SparseWindowSource:
    """Direct directory-Zarr-v3/GEFF reader; only a two-frame slab is read."""
    def __init__(self, data_root: Path, downsample: Sequence[int]) -> None:
        self.data_root = Path(data_root).resolve(strict=True)
        if not self.data_root.is_dir(): raise LaunchError("data root is not a directory")
        self.downsample = tuple(int(v) for v in downsample)
        if self.downsample != (1, 4, 4): raise LaunchError("downsample identity drift")
        self.window_stats = {"considered": 0, "usable": 0, "collision_windows": 0, "skipped_window_ids": []}
    def _contained(self, path: Path, directory: bool) -> Path:
        try: resolved = path.resolve(strict=True)
        except OSError as exc: raise LaunchError("missing source path") from exc
        try: resolved.relative_to(self.data_root)
        except ValueError as exc: raise LaunchError("source path escapes train root") from exc
        if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
            raise LaunchError("source path type mismatch")
        return resolved
    def _paths(self, record: SourceRecord) -> tuple[Path, Path]:
        zpath = self._contained(self.data_root / (record.stem + '.zarr'), True)
        gpath = self._contained(self.data_root / (record.stem + '.geff'), True)
        for path in (zpath / 'zarr.json', zpath / '0' / 'zarr.json', gpath / 'zarr.json',
                     gpath / 'nodes' / 'ids' / 'zarr.json', gpath / 'edges' / 'ids' / 'zarr.json',
                     *(gpath / 'nodes' / 'props' / k / 'values' / 'zarr.json' for k in ('t','z','y','x'))):
            self._contained(path, False)
        return zpath, gpath
    def _metadata(self, record: SourceRecord) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], tuple[float, float, float], float, float]:
        zarr = _ensure_zarr()
        zpath, gpath = self._paths(record)
        try:
            image = zarr.open_group(str(zpath), mode='r'); array = image['0']; geff = zarr.open_group(str(gpath), mode='r')
            shape = tuple(int(v) for v in array.shape); attrs = dict(image.attrs)
            meta = json.loads((gpath / 'zarr.json').read_text('utf-8')); geff_attrs = dict(geff.attrs)
            if meta.get('zarr_format') != 3 or len(shape) != 4 or min(shape) < 1: raise ValueError('bad metadata')
            multiscale = attrs['multiscales'][0]
            if (not isinstance(multiscale['axes'], list) or any(not isinstance(axis, Mapping) for axis in multiscale['axes'])
                    or tuple(str(axis.get('name', '')).lower() for axis in multiscale['axes']) != ('t','z','y','x')): raise ValueError('image axes mismatch')
            trans = multiscale['datasets'][0]['coordinateTransformations'][0]
            if trans.get('type') != 'scale' or len(trans['scale']) != 4: raise ValueError('image scale metadata mismatch')
            scale = tuple(float(v) for v in trans['scale'][1:])
            quantiles = attrs['image_statistics']['quantiles']
            qlow, qhigh = _quantile(quantiles, .001), _quantile(quantiles, .999)
            geff_meta = geff_attrs.get('geff', meta.get('attributes', {}).get('geff'))
            if not isinstance(geff_meta, Mapping) or geff_meta.get('directed') is not True: raise ValueError('GEFF must be directed')
            version = str(geff_meta.get('geff_version', ''))
            if version != '1.1' and not version.startswith('1.1.'): raise ValueError('GEFF version is incompatible')
            axes = geff_meta.get('axes')
            if (not isinstance(axes, list) or any(not isinstance(axis, Mapping) for axis in axes)
                    or tuple(str(axis.get('name', '')).lower() for axis in axes) != ('t','z','y','x')):
                raise ValueError('GEFF axes mismatch')
            geff_scale = tuple(float(axis['scale']) for axis in axes[1:])
            if any(not math.isfinite(v) or v <= 0 for v in scale + geff_scale): raise ValueError('invalid physical scale')
            if any(abs(left - right) > 1e-9 for left, right in zip(scale, geff_scale)): raise ValueError('GEFF/image scale mismatch')
            ids = np.asarray(geff['nodes/ids']); vals = {k: np.asarray(geff['nodes/props/%s/values' % k]) for k in ('t','z','y','x')}
            edgeids = np.asarray(geff['edges/ids'])
        except Exception as exc: raise LaunchError('direct GEFF/Zarr-v3 metadata decode failed') from exc
        if (qhigh <= qlow or ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer)
                or any(vals[k].ndim != 1 or len(vals[k]) != len(ids) or not np.issubdtype(vals[k].dtype, np.integer) for k in vals)):
            raise LaunchError('invalid GEFF integer rows')
        nodes = []
        for i, nid in enumerate(ids.tolist()):
            row = {'node_id': int(nid), **{k: int(vals[k][i]) for k in vals}}
            if row['node_id'] < 0 or row['t'] < 0: raise LaunchError('GEFF IDs and times must be nonnegative integers')
            nodes.append(row)
        names = getattr(edgeids.dtype, 'names', None)
        if names:
            if (edgeids.ndim != 1 or set(names) != {'source_id','target_id'}
                    or any(not np.issubdtype(edgeids.dtype.fields[k][0], np.integer) for k in ('source_id','target_id'))):
                raise LaunchError('GEFF directed edges/ids schema mismatch')
            edges = [{'source_id': int(row['source_id']), 'target_id': int(row['target_id'])} for row in edgeids]
        elif edgeids.ndim == 2 and edgeids.shape[1:] == (2,) and np.issubdtype(edgeids.dtype, np.integer):
            edges = [{'source_id': int(row[0]), 'target_id': int(row[1])} for row in edgeids]
        else: raise LaunchError('GEFF directed edges/ids schema mismatch')
        byid = {row['node_id']: row for row in nodes}
        if len(byid) != len(nodes): raise LaunchError('duplicate GEFF node')
        children: set[int] = set()
        for edge in edges:
            source, target = edge['source_id'], edge['target_id']
            if source < 0 or target < 0 or source not in byid or target not in byid or target in children:
                raise LaunchError('invalid GEFF directed graph')
            if byid[target]['t'] != byid[source]['t'] + 1: raise LaunchError('GEFF edge is not forward consecutive')
            children.add(target)
        voxel = tuple(s * d for s, d in zip(scale, self.downsample))
        return array, nodes, edges, voxel, qlow, qhigh
    def _windows(self, record: SourceRecord) -> Iterator[tuple[int, dict[str, Any]]]:
        array, raw_nodes, raw_edges, voxel, qlow, qhigh = self._metadata(record)
        raw_spatial = tuple(int(v) for v in array.shape[1:])
        spatial = tuple((limit + downsample - 1) // downsample for limit, downsample in zip(raw_spatial, self.downsample))
        byid, bytime = {}, {}
        for raw in raw_nodes:
            nid, frame = _row_int(raw, 'node_id'), _row_int(raw, 't')
            if nid in byid or frame >= int(array.shape[0]): raise LaunchError('duplicate GEFF node or time outside image')
            coordinates = {}
            for axis, downsample, raw_limit, output_limit in zip(('z','y','x'), self.downsample, raw_spatial, spatial):
                value = _row_float(raw, axis)
                if value < 0 or value >= raw_limit: raise LaunchError('GEFF coordinate outside source image')
                coordinates[axis] = min(value / downsample, float(output_limit - 1))
            row = {'id': nid, 'batch': 0, 'time': frame, **coordinates}
            byid[nid] = row; bytime.setdefault(frame, []).append(row)
        children, valid = set(), []
        for raw in raw_edges:
            source, target = _row_int(raw,'source_id'), _row_int(raw,'target_id')
            if source not in byid or target not in byid or target in children: raise LaunchError('invalid GEFF directed graph')
            if byid[target]['time'] != byid[source]['time'] + 1: raise LaunchError('GEFF edge is not forward consecutive')
            children.add(target); valid.append((source,target))
        for start in range(int(array.shape[0]) - 1):
            first, second = bytime.get(start, []), bytime.get(start+1, [])
            if not first or not second: continue
            self.window_stats['considered'] += 1
            nodes = [{**n, 'time': n['time']-start} for n in first + second]
            points = [(n['time'], *_voxel(n, spatial)) for n in nodes]
            if len(points) != len(set(points)):
                self.window_stats['collision_windows'] += 1; self.window_stats['skipped_window_ids'].append(record.group_id + ':' + str(start)); continue
            self.window_stats['usable'] += 1
            yield start, {'array': array, 'nodes': nodes, 'edges': [{'source_id':s,'target_id':t} for s,t in valid if byid[s]['time']==start and byid[t]['time']==start+1],
                'voxel_size_um': voxel, 'q_low': qlow, 'q_high': qhigh, 'spatial_shape':spatial, 'group_id':record.group_id, 'video_id':record.video_id, 'stem':record.stem}
    def validate(self, records: Sequence[SourceRecord]) -> dict[str,Any]:
        counts = {}
        for record in records:
            count = 0
            for _, meta in self._windows(record):
                for node in meta['nodes']: _voxel(node, meta['spatial_shape'])
                count += 1
            if count < 1: raise LaunchError('selected source has no usable coordinate-safe window')
            counts[record.group_id] = count
        return {'records':len(counts),'windows':sum(counts.values()),'collision_windows':self.window_stats['collision_windows']}
    def _read(self, meta: Mapping[str, Any], start: int) -> dict[str, Any]:
        raw = np.asarray(meta['array'][start:start+2, ::1, ::4, ::4], dtype=np.float32)
        if raw.shape[0] != 2 or tuple(raw.shape[1:]) != tuple(meta['spatial_shape']) or not np.isfinite(raw).all(): raise LaunchError('two-frame Zarr slab read failed')
        frames = torch.from_numpy(np.clip((raw-float(meta['q_low']))/(float(meta['q_high'])-float(meta['q_low'])),0,1)).unsqueeze(0).unsqueeze(2)
        return {'frames':frames, 'nodes':meta['nodes'], 'edges':meta['edges'], 'voxel_size_um':meta['voxel_size_um'], 'window_id':meta['group_id']+':'+str(start)}
    def batches(self, records: Sequence[SourceRecord], *, completed_windows: int, deadline: float, epochs: int=1) -> Iterator[dict[str, Any]]:
        position = 0; usable = False
        for epoch in range(1, epochs+1):
            for record in records:
                for start, meta in self._windows(record):
                    usable = True
                    if time.monotonic() >= deadline: return
                    if position >= completed_windows:
                        batch = self._read(meta,start); batch['epoch'] = epoch; yield batch
                    position += 1
        if not usable: raise LaunchError('no usable source windows remain')

def _build_released_model(args: argparse.Namespace, device: torch.device) -> TemporalAdapterModel:
    config = EXPECTED_CONFIG
    backbone = TemporalUNet3D(in_channels=1, out_channels=config["unet_out_channels"], layers=config["unet_layers"])
    detector, _ = load_released_core(backbone, args.weights_path, args.config_path)
    model = TemporalAdapterModel(backbone, detector)
    model.set_unfreeze_depth(args.unfreeze)
    return model.to(device)


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _finite_state(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return _finite_tensor(value)
    if isinstance(value, Mapping):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _batch_loss(model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device) -> Mapping[str, torch.Tensor]:
    frames = batch["frames"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    output = model(frames, scenario="universal")
    targets = build_sparse_targets(output, batch["nodes"], batch["edges"], batch["voxel_size_um"])
    return multitask_loss(output, targets, division_enabled=False)


def train_steps(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batches: Iterable[Mapping[str, Any]], *,
                device: torch.device, deadline: float, max_steps: int,
                scaler: Any | None = None,
                checkpoint_callback: Callable[[Mapping[str, Any], Any | None], None] | None = None) -> tuple[dict[str, Any], Any | None]:
    """Execute real forward/backward/step work, checking cap before each step."""
    if max_steps < 0:
        raise LaunchError("remaining steps cannot be negative")
    enabled = USE_MIXED_PRECISION and device.type == "cuda"
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=enabled)
    sums = {"total": 0.0, "center": 0.0, "displacement": 0.0, "log_variance": 0.0, "division": 0.0}
    by_epoch: dict[int, dict[str, Any]] = {}
    completed, window_ids, per_step, status = 0, [], [], "completed"
    model.train()
    for batch in batches:
        if completed >= max_steps:
            break
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=enabled):
            losses = _batch_loss(model, batch, device)
        invalid_losses = [name for name, value in losses.items() if not _finite_tensor(value)]
        if invalid_losses:
            raise LaunchError("nonfinite loss components: " + ",".join(invalid_losses))
        loss = losses["total"]
        if enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        invalid_gradients = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None and not _finite_tensor(parameter.grad)]
        if invalid_gradients:
            raise LaunchError("nonfinite gradients: " + ",".join(invalid_gradients[:8]))
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        if time.monotonic() >= deadline:
            status = "time_cap"
            break
        if enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if not all(_finite_tensor(parameter.detach()) for parameter in model.parameters()):
            raise LaunchError("nonfinite parameter after optimizer step")
        completed += 1
        window_ids.append(str(batch.get("window_id", completed)))
        epoch = int(batch.get("epoch", 1))
        if epoch < 1:
            raise LaunchError("batch epoch must be positive")
        aggregate = by_epoch.setdefault(epoch, {"steps": 0, "sums": {key: 0.0 for key in sums}})
        aggregate["steps"] += 1
        for key in sums:
            sums[key] += float(losses[key].detach().cpu())
            aggregate["sums"][key] += float(losses[key].detach().cpu())
        per_step.append({"step": completed, "epoch": epoch, "window_id": window_ids[-1],
                         "loss_components": {key: float(losses[key].detach().cpu()) for key in sums}})
        if checkpoint_callback is not None and completed % CHECKPOINT_EVERY_STEPS == 0:
            checkpoint_callback(_step_result("running", completed, window_ids, per_step, sums, by_epoch), scaler)
    return _step_result(status, completed, window_ids, per_step, sums, by_epoch), scaler


def _step_result(status: str, completed: int, window_ids: Sequence[str], per_step: Sequence[Mapping[str, Any]],
                 sums: Mapping[str, float], by_epoch: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {key: value / max(1, completed) for key, value in sums.items()}
    epochs = [{"epoch": epoch, "steps": item["steps"],
               "loss_components": {key: value / item["steps"] for key, value in item["sums"].items()}}
              for epoch, item in sorted(by_epoch.items())]
    return {"status": status, "steps": completed, "windows": list(window_ids), "per_step": list(per_step),
            "epoch_metrics": epochs, "loss_components": aggregate}


def _load_resume(path: Path, identity: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scaler: Any | None) -> tuple[int, float, Any | None]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LaunchError("atomic checkpoint weights_only load failed") from exc
    if not isinstance(state, dict) or state.get("identity") != dict(identity) or not _finite_state(state):
        raise LaunchError("resume identity or finite-state check failed")
    try:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        steps, elapsed = int(state["completed_steps"]), float(state["elapsed_seconds"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchError("resume state is incompatible") from exc
    if steps < 0 or not math.isfinite(elapsed) or elapsed < 0:
        raise LaunchError("resume progress is invalid")
    return steps, elapsed, scaler


def _write_outputs(output_dir: Path, identity: Mapping[str, Any], model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, scaler: Any | None, result: Mapping[str, Any],
                   started: float, source_records: Sequence[SourceRecord], prior_steps: int, prior_elapsed: float,
                   epochs: int, max_steps: int) -> dict[str, Any]:
    elapsed = prior_elapsed + time.monotonic() - started
    total_steps = prior_steps + int(result["steps"])
    state = {"identity": dict(identity), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None, "completed_steps": total_steps,
             "elapsed_seconds": elapsed, "loss_components": dict(result["loss_components"]),
             "source_groups": [record.group_id for record in source_records],
             "source_video_ids": [record.video_id for record in source_records],
             "source_records": [record.json() for record in source_records], "completed_windows": total_steps,
             "execution": {"epochs": epochs, "max_steps": max_steps}, "status": result["status"]}
    checkpoint_sha256 = _atomic_checkpoint(output_dir / "checkpoint.pt", state)
    metrics = {"identity": dict(identity), "checkpoint_sha256": checkpoint_sha256, "completed_steps": total_steps,
               "elapsed_seconds": elapsed, "loss_components": dict(result["loss_components"]),
               "source_groups": [record.group_id for record in source_records],
               "source_video_ids": [record.video_id for record in source_records],
               "source_records": [record.json() for record in source_records], "windows": list(result["windows"]),
               "per_step": list(result.get("per_step", [])), "epoch_metrics": list(result.get("epoch_metrics", [])),
               "source_window_stats": dict(result.get("source_window_stats", {})),
               "execution": {"epochs": epochs, "max_steps": max_steps},
               "status": result["status"]}
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(output_dir / "resume.json", {"identity": dict(identity), "checkpoint_sha256": checkpoint_sha256,
                 "completed_steps": total_steps, "source_groups": [record.group_id for record in source_records],
                 "source_video_ids": [record.video_id for record in source_records],
                 "execution": {"epochs": epochs, "max_steps": max_steps}, "status": result["status"]})
    return metrics


def _run_bounded_training(args: argparse.Namespace, selected: Sequence[SourceRecord], device_name: str,
                          model_factory: Callable[[argparse.Namespace, torch.device], torch.nn.Module] | None = None,
                          batch_factory: Callable[[int, float], Iterable[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    _validate_arg_contract(args)
    if not args.armed:
        raise LaunchError("non-dry launch requires explicit --armed before data access")
    if args.data_root is None and batch_factory is None:
        raise LaunchError("--data-root is required when dry-run is disabled")
    if args.scenario != "universal":
        raise LaunchError("specialist training is deferred until T-05")
    device = torch.device(device_name)
    identity = {**_identity(args), "selected_sources": [record.json() for record in selected]}
    _seed_all(args.seed)
    started = float(getattr(args, "started_monotonic", time.monotonic()))
    if not math.isfinite(started):
        raise LaunchError("end-to-end start time is invalid")
    deadline = started + max(0.0, args.maximum_hours * 3600.0)
    model = (model_factory or _build_released_model)(args, device)
    detector = getattr(model, "detect_head", None)
    if detector is not None and any(parameter.requires_grad for parameter in detector.parameters()):
        raise LaunchError("released detector must remain frozen for sparse-label training")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise LaunchError("no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler: Any | None = torch.amp.GradScaler("cuda", enabled=USE_MIXED_PRECISION and device.type == "cuda")
    prior_steps, prior_elapsed = 0, 0.0
    if args.resume is not None:
        prior_steps, prior_elapsed, scaler = _load_resume(args.resume, identity, model, optimizer, scaler)
    deadline -= prior_elapsed
    if prior_steps >= args.max_steps:
        empty = {key: 0.0 for key in ("total", "center", "displacement", "log_variance", "division")}
        result = {"status": "completed", "steps": 0, "windows": [], "per_step": [],
                  "epoch_metrics": [], "loss_components": empty, "source_window_stats": {}}
    elif time.monotonic() >= deadline:
        empty = {key: 0.0 for key in ("total", "center", "displacement", "log_variance", "division")}
        result = {"status": "time_cap", "steps": 0, "windows": [], "per_step": [],
                  "epoch_metrics": [{"epoch": 1, "steps": 0, "loss_components": empty}], "loss_components": empty}
    else:
        source: SparseWindowSource | None = None
        if batch_factory is not None:
            batches = batch_factory(prior_steps, deadline)
        else:
            source = SparseWindowSource(args.data_root, EXPECTED_CONFIG["downsample"])
            batches = source.batches(selected, completed_windows=prior_steps, deadline=deadline, epochs=args.epochs)
        def checkpoint_callback(progress: Mapping[str, Any], progress_scaler: Any | None) -> None:
            metrics = _write_outputs(args.output_dir, identity, model, optimizer, progress_scaler, progress, started,
                                     selected, prior_steps, prior_elapsed, args.epochs, args.max_steps)
            print(json.dumps({"status": "running", "steps": metrics["completed_steps"], "output": str(args.output_dir)},
                             sort_keys=True), flush=True)
        result, scaler = train_steps(model, optimizer, batches, device=device,
                                     deadline=deadline, max_steps=args.max_steps - prior_steps, scaler=scaler,
                                     checkpoint_callback=checkpoint_callback)
        result["source_window_stats"] = dict(getattr(source, "window_stats", {})) if source is not None else {}
    return _write_outputs(args.output_dir, identity, model, optimizer, scaler, result, started, selected, prior_steps,
                          prior_elapsed, args.epochs, args.max_steps)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)



KROOT = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
KALT = Path('/kaggle/input/biohub-cell-tracking-during-development/train')
# The released checkpoint is a separate immutable Kaggle input. The support
# input is used only for offline wheels; do not walk either input tree.
CHECKPOINT_DIRS = (
    Path('/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts/weights/unet_transformer/split_0'),
    Path('/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts/cellmot-baseline-artifacts/weights/unet_transformer/split_0'),
    Path('/kaggle/input/cellmot-baseline-artifacts/weights/unet_transformer/split_0'),
    Path('/kaggle/input/cellmot-baseline-artifacts/cellmot-baseline-artifacts/weights/unet_transformer/split_0'),
)
KOUT = Path('/kaggle/working/t03')
ARG_FIELDS = ('dry_run','fold','output_dir','resume','armed','scenario','unfreeze','lr','fraction','seed','epochs','max_steps','maximum_hours','maximum_charge')
def _validate_arg_contract(args: argparse.Namespace) -> None:
    missing = [name for name in ARG_FIELDS if not hasattr(args,name)]
    if missing: raise LaunchError('argument namespace is incomplete: ' + ','.join(missing))
    if not isinstance(args.dry_run,bool) or not isinstance(args.armed,bool) or args.armed != (not args.dry_run): raise LaunchError('armed argument contract mismatch')
    if not isinstance(args.output_dir,Path) or (args.resume is not None and not isinstance(args.resume,Path)): raise LaunchError('path argument contract mismatch')
    if args.fold not in (0,1,2) or args.scenario != 'universal' or args.unfreeze != 0: raise LaunchError('candidate argument contract mismatch')
    numeric = (args.lr,args.fraction,args.maximum_hours,args.maximum_charge)
    if not all(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value)) for value in numeric): raise LaunchError('numeric argument contract mismatch')
    if not all(isinstance(value,int) and not isinstance(value,bool) for value in (args.seed,args.epochs,args.max_steps)): raise LaunchError('integer argument contract mismatch')
def _runtime_preflight(args: argparse.Namespace) -> dict[str,Any]:
    _validate_arg_contract(args)
    if args.dry_run: return {'mode':'dry','dependency_preflight':False}
    if not torch.cuda.is_available(): raise LaunchError('CUDA is required for an armed Kaggle run')
    if not hasattr(torch,'amp') or not hasattr(torch.amp,'GradScaler') or not hasattr(torch,'optim'): raise LaunchError('Kaggle torch runtime lacks required training APIs')
    try:
        if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'): raise LaunchError('Kaggle numpy runtime is below zarr minimum')
    except (AttributeError,ValueError) as exc: raise LaunchError('Kaggle numpy version cannot be validated') from exc
    _ensure_zarr()
    ready = ZARR_DEPS / '.ready'
    if not ready.is_file() or not re.fullmatch(r'[0-9a-f]{64}\n',ready.read_text('ascii')): raise LaunchError('offline dependency ready marker is invalid')
    return {'mode':'armed','dependency_preflight':True,'manual_wheel_count':len(ZARR_WHEELS),'native_import_count':len(RUNTIME_NATIVE_IMPORTS),'numpy':np.__version__,'torch':torch.__version__,'ready_sha256':ready.read_text('ascii').strip(),'versions':{name:importlib.metadata.version(name) for name in sorted(RUNTIME_DISTS)}}
def _exact_candidate(args: argparse.Namespace) -> None:
    trial = trial_record(); c = trial['candidate']
    got = {'fold':args.fold,'scenario':args.scenario,'unfreeze':args.unfreeze,'learning_rate':args.lr,'fit_fraction':args.fraction,'seed':args.seed,'epochs':args.epochs,'max_steps':args.max_steps,'batch_size':1,'division_enabled':False,'center_supervision':CENTER_SUPERVISION,'center_loss':CENTER_LOSS,'released_detector_frozen':True}
    if got != c or args.maximum_hours != .75 or args.maximum_charge != .30: raise LaunchError('armed candidate identity or cap drift')
    immutable = trial['immutable_inputs']
    if (immutable['registry_sha256'] != REGISTRY_SHA256 or immutable['registry_projection_sha256'] != REGISTRY_PROJECTION_SHA256): raise LaunchError('trial registry identity drift')
    if immutable['checkpoint_sha256'] != WEIGHTS_SHA256 or immutable['checkpoint_config_sha256'] != CONFIG_SHA256: raise LaunchError('trial checkpoint identity drift')
def _input_entries(root: Path=Path('/kaggle/input'), cap: int=32) -> list[str]:
    if cap <= 0: raise LaunchError('input diagnostic cap must be positive')
    try:
        if not root.is_dir(): return []
        return sorted(item.name for item in root.iterdir())[:cap]
    except OSError as exc:
        return [f'<unreadable:{type(exc).__name__}>']

def _find_input(name: str, size: int, digest: str, diagnostic_root: Path=Path('/kaggle/input')) -> Path:
    hits = []
    probes = []
    for directory in CHECKPOINT_DIRS:
        path = directory / name
        probe: dict[str, Any] = {'path':str(path),'exists':path.is_file()}
        if not probe['exists']:
            probes.append(probe)
            continue
        try: observed_size=path.stat().st_size
        except OSError as exc:
            probe['stat_error']=type(exc).__name__; probes.append(probe); continue
        probe['bytes']=observed_size
        if observed_size != size:
            probes.append(probe)
            continue
        try: observed_digest=_sha_file(path)
        except OSError as exc:
            probe['hash_error']=type(exc).__name__; probes.append(probe); continue
        probe['sha256']=observed_digest
        probes.append(probe)
        if observed_digest == digest:
            hits.append(path)
    if len(hits) != 1:
        diagnostic={'expected_bytes':size,'expected_sha256':digest,'input_entries':_input_entries(diagnostic_root),'probes':probes}
        raise LaunchError('checkpoint exact-match search must yield one file: '+json.dumps(diagnostic,sort_keys=True,separators=(',',':')))
    return hits[0]
def _sibling_config(weights: Path) -> Path:
    path = Path(weights).with_name('config.json')
    if not path.is_file() or path.stat().st_size != CONFIG_BYTES or _sha_file(path) != CONFIG_SHA256: raise LaunchError('checkpoint sibling config identity mismatch')
    return path
def _args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description='private first-candidate trainer')
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--fold',type=int,choices=(0,1,2),default=0)
    p.add_argument('--output-dir',type=Path,default=KOUT); p.add_argument('--resume',type=Path)
    a=p.parse_args(argv); a.armed=not a.dry_run; a.scenario='universal'; a.unfreeze=0; a.lr=1e-5; a.fraction=.1; a.seed=17; a.epochs=1; a.max_steps=1000; a.maximum_hours=.75; a.maximum_charge=.30
    _validate_arg_contract(a); return a
def main(argv: Sequence[str] | None=None) -> int:
    args=_args(argv); args.started_monotonic=time.monotonic(); _validate_budget(args.maximum_hours,args.maximum_charge); runtime=_runtime_preflight(args)
    split, provenance = embedded_manifest('split'), embedded_manifest('provenance'); registry = runtime_registry_projection()
    selected_groups=select_fit_groups(args.fold,split,provenance,args.fraction,args.seed)
    _, forbidden=_fold_data(args.fold,split,provenance); selected=resolve_fit_records(selected_groups,registry,forbidden)
    if not args.dry_run:
        _exact_candidate(args)
        if args.output_dir.resolve() != KOUT: raise LaunchError('armed output root must be private Kaggle working/t03')
        args.data_root=KROOT if KROOT.is_dir() else KALT
        if not args.data_root.is_dir(): raise LaunchError('competition train root is absent')
        args.weights_path=_find_input('edge_predictor_best.pth',WEIGHTS_BYTES,WEIGHTS_SHA256); args.config_path=_sibling_config(args.weights_path)
        verify_released_files(args.weights_path,args.config_path)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pre={'artifact':'prelaunch-v1','status':'not_launched' if args.dry_run else 'armed','identity':{**_identity(args),'selected_sources':[r.json() for r in selected]},'source_isolation':{'fit_only':True,'selected_group_ids':[r.group_id for r in selected],'selected_train_video_ids':[r.video_id for r in selected],'held_out_used':False,'audit_used':False,'test_used':False},'execution':{'device':'cuda' if torch.cuda.is_available() else 'cpu','mixed_precision':USE_MIXED_PRECISION,'gradient_clip_norm':1.0,'epochs':1,'max_steps':1000,'batch_size':1,'dry_run':args.dry_run,'cap_scope':'notebook-start-through-checkpoint'},'training_policy':{'center_supervision':CENTER_SUPERVISION,'center_loss':CENTER_LOSS,'released_detector_frozen':True,'division_enabled':False},'runtime_preflight':runtime,'spend_gate':{'maximum_hours':.75,'maximum_charge':.30,'cap_hours':.75,'cap_charge':.30,'approved_spend':not args.dry_run}}
    digest=_atomic_json(args.output_dir/'prelaunch.json',pre)
    if args.dry_run: print(json.dumps({'status':'dry','prelaunch_sha256':digest,'sources':len(selected)},sort_keys=True)); return 0
    print(json.dumps({'status':'armed','output':str(KOUT),'sources':len(selected)},sort_keys=True),flush=True)
    result=_run_bounded_training(args,selected,'cuda')
    print(json.dumps({'status':result['status'],'steps':result['completed_steps'],'output':str(KOUT)},sort_keys=True),flush=True); return 0
if __name__ == '__main__': raise SystemExit(main([]))
