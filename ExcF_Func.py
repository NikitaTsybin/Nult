#import scipy.integrate as integrate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
from math import cos, sqrt, acos, floor

Tmax = 500
d = 12/1000
fi1 = 0.62
fi2 = 0.5
lmbd = (1.2 - 0.00035*450)
C = (0.71 + 0.00083*450)
ared = 3.6*lmbd/((C+ 0.05*2.5)*2350)


#Глубина прогрева
#t - минуты; результат - метры
def l_pr(t):
##    return (12*ared*t/60)**0.5
    return (0.2*ared*t)**0.5

#Глубина прогрева до критической температуры Tmax
#t - минуты; результат - метры
def a_t (t):
##    at = ( (t/1)**0.5*( (1/5)**0.5 - ((Tmax-20)/6000)**0.5) - fi1 ) * ared**0.5
    r1 = 1 - ((Tmax-20)/1200)**0.5
    at = r1*l_pr(t) - fi1*ared**0.5
    return at

#Распределение температур в бетоне
#t - минуты; x - метры; результат - градусы
def Tb (x, t):
    rb = (x + fi1*ared**0.5)/l_pr(t)
    Tb = 20
    if rb<1:
        Tb = 20 + 1200*(1 - rb)**2
    return Tb

def Ta (x, t, d):
    ra = (x - d/2 + fi1*ared**0.5 + fi2*d)/l_pr(t)
##    print(ra*l_pr(t))
##    print(ra)
    Ta = 20
    if ra<1:
        Ta = 20 + 1200*(1 - ra)**2
    return Ta

#Функция возвращает ближайшие к числу индексы и значения массива
def take_nearest(num,arr):
    temp = min(arr,key=lambda x:abs(x-num))
    ind = arr.index(temp)
    if temp<num: a = ind
    else: a = ind - 1
    return (a, a+1), (arr[a], arr[a+1])

#Функция линейной интерполяции d = [[x1, y1], [x2, y2]]
#Возвращает значение линейной интерполяции y при x
def l_interp(d1, x):
    rez = d1[0][1] + (x - d1[0][0]) * ((d1[1][1] - d1[0][1])/(d1[1][0] - d1[0][0]))
    return(rez)

#Функция строит линейную интерполяцию по массиву
def l_interp_list (x, x_arr, y_arr):
    rng = take_nearest(x, x_arr)
    y1 = y_arr[rng[0][0]]
    y2 = y_arr[rng[0][1]]
    x1 = rng[1][0]
    x2 = rng[1][1]
    d1 = [[x1, y1], [x2, y2]]
    y = l_interp(d1, x)
    return y

arr_T1 = [0, 20, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1500] #Массив температур бетона
arr_gbt = [1.0, 1.0, 0.98, 0.95, 0.85, 0.8, 0.6, 0.2, 0.1, 0, 0, 0] #Массив понижающих коэфф. к прочности бетона
arr_bbt = [1.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0, 0, 0] #Массив понижающих коэфф. к модулю бетона
arr_T2 = [0, 50, 100, 300, 500, 700, 1500] #Массив температур бетона
arr_abt = [9*10**(-6), 9*10**(-6), 9*10**(-6), 8*10**(-6), 11*10**(-6), 14.5*10**(-6), 14.5*10**(-6)]
arr_T3 = [0, 20, 100, 200, 300, 400, 500, 600, 700, 800, 1500] #Массив температур арматуры
arr_ast = [11.5*10**(-6), 11.5*10**(-6), 12*10**(-6), 12.5*10**(-6), 13*10**(-6), 13.5*10**(-6),
           14.0*10**(-6), 14.5*10**(-6), 15.0*10**(-6), 15.5*10**(-6), 15.5*10**(-6)]
arr_T4 = [0, 20, 200, 300, 400, 500, 600, 700, 800, 850, 1500] #Массив температур арматуры
arr_gst = [1.0, 1.0, 1.0, 1.0, 0.85, 0.6, 0.37, 0.22, 0.1, 0, 0] #Массив понижающих коэфф. к прочности арматуры
arr_bst = [1.0, 1.0, 0.92, 0.9, 0.85, 0.8, 0.77, 0.72, 0.65, 0, 0] #Массив понижающих коэфф. к модулю арматуры


def gammabt (T):
    return l_interp_list(T, arr_T1, arr_gbt)

def gammast (T):
    return l_interp_list(T, arr_T4, arr_gst)

def betabt (T):
    return l_interp_list(T, arr_T1, arr_bbt)

def betast (T):
    return l_interp_list(T, arr_T4, arr_bst)

def alphab (T):
    return l_interp_list(T, arr_T2, arr_abt)

def alphas (T):
    return l_interp_list(T, arr_T3, arr_ast)

def plot_gammabt():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    gbt = np.array([gammabt(i) for i in x])
    fig.add_trace(go.Scatter(x=x, y=gbt))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[-0.01,1.0])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_gammast():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    gbt = np.array([gammast(i) for i in x])
    fig.add_trace(go.Scatter(x=x, y=gbt))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[-0.01,1.0])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_alphab():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    gbt = np.array([alphab(i)*10**6 for i in x])
    fig.add_trace(go.Scatter(x=x, y=gbt))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[6,16])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_alphas():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    gbt = np.array([alphas(i)*10**6 for i in x])
    fig.add_trace(go.Scatter(x=x, y=gbt))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[10,16])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig


def plot_betabt():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    bbt = np.array([betabt(i) for i in x])
    fig.add_trace(go.Scatter(x=x, y=bbt))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[-0.01,1.0])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_betast():
    fig = go.Figure()
    x = np.arange(0, 1200, 10)
    bst = np.array([betast(i) for i in x])
    fig.add_trace(go.Scatter(x=x, y=bst))
    fig.update_xaxes(title="T")
    fig.update_xaxes(range=[0,1200])
    fig.update_yaxes(range=[-0.01,1.0])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_lpr():
    fig = go.Figure()
    x = np.arange(30, 300, 5)
    lpr = np.array([l_pr(i)*1000 for i in x])
    fig.add_trace(go.Scatter(x=x, y=lpr))
    fig.update_xaxes(title="Время, мин")
    fig.update_xaxes(range=[30,240])
    fig.update_yaxes(range=[50,260])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_at():
    fig = go.Figure()
    x = np.arange(30, 300, 5)
    at = np.array([a_t(i)*1000 for i in x])
    fig.add_trace(go.Scatter(x=x, y=at))
    fig.update_xaxes(title="Время, мин")
    fig.update_xaxes(range=[30,240])
    fig.update_yaxes(range=[0,75])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=200)
    return fig

def plot_T(h,tau):
    fig = go.Figure()
    x_crit = a_t(tau)*1000
    y_crit = Tb(x_crit/1000, tau)
    xx = np.arange(0, h*10, 1)
    T = []
    for i in xx:
        T.append(Tb(i/1000, tau))
    T = np.array(T)
    fig.add_trace(go.Scatter(x=xx, y=T, showlegend=False))
    xx = [0, x_crit, x_crit, 0, 0]
    yy = [0, 0, y_crit, y_crit, 0]
    fig.add_trace(go.Scatter(x=xx, y=yy, fill='tozeroy', fillcolor="darkred", showlegend=False, mode='lines', line=dict(color="#0000ff")))
    fig.update_xaxes(title="Толщина, мм")
    fig.update_xaxes(range=[0,h*10])
    fig.update_yaxes(range=[0,1200])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(height=300)
    return fig



#### ОСНОВНЫЕ РАСХОЖДЕНИЯ С СП
#### КОЭФФИЦИЕНТ ДЕЛЬТА Е (СП63.13330.2018 П. 8.1.15 МИНИМАЛЬНОЕ 0.15)
#### УЧИТЫВАТЬ ЛИ РАЗНЫЕ МОДУЛИ ДЛЯ СЖАТОЙ И РАСТЯНУТОЙ ПРИ ОПРЕДЕЛЕНИИ ЖЕСТКОСТИ
#### МОМЕНТ ИНЕРЦИИ БЕТОНА С УЧЕТОМ ПРОГРЕТОЙ ЗОНЫ?


def solve_NF (b, h, L, a, mu, kdl, As, Asc, Eb0, Es, xiR, Rsn, Rsc, Rbn, tau, d, is_et, d_e_min_sp, I_b_sp):
    d_e_min = 0.3
    if d_e_min_sp:
        d_e_min = 0.15
    ac = a #Привязка сжатой арматуры, см
    L0 = L*mu #Расчетная длина в см
    gz = 9.80665 #Ускорение свободного падения
    gz = 10 #Ускорение свободного падения
    pi = 3.1415927 #Число Пи
    h0 = h - a #Рабочая высота сечения, см
    at = a_t(tau)*100 #Глубина прогрева до критической температуры, см
##    print('at=',at)
    h0t = h0 - at #Рабочая высота сечения за вычетом бетоны горячее 500, см
    Ts = Ta((h-a)/100, tau, d) #Температура растянутой арматуры
    Rsnt = Rsn*gammast(Ts) #Расчетное сопротивление растянутой арматуры
    Estt = Es*betast(Ts) #Модуль растянутой арматуры
    Tsc = Ta(a/100, tau, d/100) #Температура сжатой арматуры
    Rsct = Rsc*gammast(Tsc) #Расчетное сопротивление сжатой арматуры
    Esct = Es*betast(Tsc) #Модуль сжатой арматуры
    aa = 0.7
    et = 0
    if is_et:
        Tbt = Tb((h)/100, tau)  #Температура необогреваемой грани бетона
        alst = alphas(Tsc)
        albt = alphab(Tbt)
        et = aa*(alst*Tsc - albt*Tbt)*L0**2/8/h0t
##        print('et=',et)
##    print('321', betast(Tsc),Es,Esct)
    epssel = Rsnt/Estt
    epsb2 = 0.0035
    xi_R = 0.8 / (1 + epssel / epsb2)
    Tb0 = Tb((h/2)/100, tau)
##    print('xir=',xi_R)
    gammabt(Tb0)
    Rbnt = Rbn*1
    Ebt = Eb0*betabt(Tb0)
##    print('betab', betabt(Tb0), 'Eb0', Eb0, 'Etb', Ebt)
    #Момент инерции бетона
    I = (b*(h-at)**3)/12
    if I_b_sp:
        I = (b*(h)**3)/12
    #Момент инерции арматуры
    Ist = As*(h/2 - a)**2
    Isc = Asc*(h/2 - ac)**2
    #Коэффициент, учитывающий влияние длительности
    fi_dl = 1 + kdl
    fi_dl = 2
    if fi_dl>2:
        fi_dl = 2
    #Предполагаем, что начальный эксцентриситет из статического расчета
    #не превышает случайный
    #Случайный эксцентриситет
    e0 = max(1.0, h/30, L/600)
    #Относительное значение эксцентриситета
    d_e = e0 / h
    if d_e<d_e_min:
        d_e = d_e_min
    if d_e>1.5:
        d_e = 1.5
    #Коэффициенты жесткости
    kb = 0.15/(fi_dl*(0.3+d_e))
    ks = 0.7
##    print('kb=',kb, 'ks=',ks)
    #Жескость железобетонного сечения
##    D = kb*Ebt*I + ks*(Esct*Isc + Estt*Ist)
    D = kb*Ebt*I + ks*(Esct*Isc + Esct*Ist)
    #Условная критическая сила
    Ncr = pi**2*D/(L0**2)
##    print('kb=', kb, 'ks', ks)
##    print('L0=', L0, 'Ncr=', Ncr)
    #Вычисление вспомогательных величин
##    print('et=', et)
    F1 = Rsnt*As*(1+xi_R)/(1-xi_R) - Rsct*Asc
    F2 = Rbnt*b + 2*Rsnt*As / (h0t*(1-xi_R))
    beta_1 = (h0 - ac)*F2**2 / (Rbnt*b)
    beta_2 = F1 - h0t*F2
    beta_3 = F1*F2*h0t - F1**2/2
    A = beta_1 + 2*beta_2 - Ncr + (2*et*F2**2 / (Rbnt*b))
    B = ((2*e0 + h0 - ac)/(h0-ac)*beta_1 + 2*beta_2)*Ncr + 2*Rsct*Asc*beta_1 + 2*beta_3 + (2*et*Ncr*F2**2 / (Rbnt*b))
    C = 2*(beta_3 + Rsct*Asc*beta_1)*Ncr
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
    check_xi = (N + F1)/(F2 * (h0t))
    if check_xi < xi_R:
            print('Упали')
            F1 = Rsnt*As - Rsct*Asc
            F2 = Rbnt*b
            beta_1 = (h0 - ac)*F2**2 / (Rbnt*b)
            beta_2 = F1 - h0t*F2
            beta_3 = F1*F2*h0t - F1**2/2
            A = beta_1 + 2*beta_2 - Ncr + (2*et*F2**2 / (Rbnt*b))
            B = ((2*e0 + h0 - ac)/(h0-ac)*beta_1 + 2*beta_2)*Ncr + 2*Rsct*Asc*beta_1 + 2*beta_3 + (2*et*Ncr*F2**2 / (Rbnt*b))
            C = 2*(beta_3 + Rsct*Asc*beta_1)*Ncr
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
##    print('После N=',round(N/gz*100*100))
    sigmakN = N/b/h
    sigmatf = sigmakN/gz*100*100
    return N, sigmakN, sigmatf
##    return round(Ts), round(Rsnt*10), round(Estt*10), round(Tsc), round(Rsct*10), round(Esct*10), round(Tb0), round(Rbnt*10), round(Ebt*10)


