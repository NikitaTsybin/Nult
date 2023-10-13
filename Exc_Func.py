import math
from math import cos, sqrt, acos, floor


def solve_N (b, h, L, a, mu, kdl, As, Asc, Eb0, Es, xi_R, Rs, Rsc, Rbc):
    gz = 9.80665 #Ускорение свободного падения
    gz = 10 #Ускорение свободного падения
    pi = 3.1415927 #Число Пи
    h0 = h - a
    #Расчетная длина
    L0 = L*mu
    #Привязка сжатой арматуры
    ac = a
    #Момент инерции бетона
    I = (b*h**3)/12
    #Момент инерции арматуры
    Is = As*(h/2 - a)**2 + Asc*(h/2 - ac)**2
    #Коэффициент, учитывающий влияние длительности
    fi_dl = 1 + kdl
    if fi_dl>2:
        fi_dl = 2
    #Предполагаем, что начальный эксцентриситет из статического расчета
    #не превышает случайный
    #Случайный эксцентриситет
    e0 = max(1.0, h/30, L/600)
    #Относительное значение эксцентриситета
    d_e = e0 / h
    if d_e<0.15:
        d_e = 0.15
    if d_e>1.5:
        d_e = 1.5
    #Коэффициенты жесткости
    kb = 0.15/(fi_dl*(0.3+d_e))
    ks = 0.7
    #Жескость железобетонного сечения
    D = kb*Eb0*I + ks*Es*Is
    #Условная критическая сила
    Ncr = pi**2*D/(L0**2)
    #Вычисление вспомогательных величин
    F1 = Rs*As*(1+xi_R)/(1-xi_R) - Rsc*Asc
    F2 = Rbc*b + 2*Rs*As/( h0*(1-xi_R) )
    beta_1 = (h0 - ac)*F2**2 / (Rbc*b)
    beta_2 = (F1 - F2*h0)
    beta_3 = F1*F2*h0 - F1**2/2
    A = (beta_1 + 2*beta_2 - Ncr)
    B = 2*( (e0/(h0-ac) + 1/2)*Ncr*beta_1 + Ncr*beta_2 + beta_3 + Rsc*Asc*beta_1 )
    C = 2*(beta_3 + Rsc*Asc*beta_1)*Ncr
    Q = (A**2 + 3*B)/9
    R = (2*A**3 + 9*A*B + 27*C)/54
    S = Q**3 - R**2
    phi = acos(R/sqrt(Q**3))/3
    x1 = -2*sqrt(Q)*cos(phi) - A/3
    x2 = -2*sqrt(Q)*cos(phi + 2*pi/3) - A/3
    x3 = -2*sqrt(Q)*cos(phi - 2*pi/3) - A/3
    xmin = min(x1, x2, x3)
    xmid = min(max(x1,x2), max(x2,x3), max(x3,x1))
    xmax = max(x1, x2, x3)
    N = xmid
    if N<0:
        N = xmax
    check_xi = (N + F1)/(F2 * h0)
    if check_xi < xi_R:
            F1 = Rs*As - Rsc*Asc
            F2 = Rbc*b
            beta_1 = (h0 - ac)*F2**2 / (Rbc*b)
            beta_2 = (F1 - F2*h0)
            beta_3 = F1*F2*h0 - F1**2/2
            A = (beta_1 + 2*beta_2 - Ncr)
            B = 2*( (e0/(h0-ac) + 1/2)*Ncr*beta_1 + Ncr*beta_2 + beta_3 + Rsc*Asc*beta_1 )
            C = 2*(beta_3 + Rsc*Asc*beta_1)*Ncr
            Q = (A**2 + 3*B)/9
            R = (2*A**3 + 9*A*B + 27*C)/54
            S = Q**3 - R**2
            phi = acos(R/sqrt(Q**3))/3
            x1 = -2*sqrt(Q)*cos(phi) - A/3
            x2 = -2*sqrt(Q)*cos(phi + 2*pi/3) - A/3
            x3 = -2*sqrt(Q)*cos(phi - 2*pi/3) - A/3
            xmin = min(x1, x2, x3)
            xmid = min(max(x1,x2), max(x2,x3), max(x3,x1))
            xmax = max(x1, x2, x3)
            N = xmid
            if N<0:
                N = xmax
    sigmakN = N/b/h
    sigmatf = sigmakN/gz*100*100
    return N, sigmakN, sigmatf
