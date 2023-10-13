import streamlit as st
import math
import re
from math import cos, sqrt, acos, floor
import pandas as pd
from Reinforcement_and_Concretes import *
from Area_from_string import *
from Exc_Func import *
from ExcF_Func import *

#Данные по умолчанию
pi = 3.1415927 #Число Пи
gz = 9.80665 #Ускорение свободного падения
gz = 10 #Ускорение свободного падения
coeff_b_init = 0.85 #Понижающий коэффициент к прочности бетона
b_init = 100 #Начальная ширина сечения
h_init = 25 #Начальная высота сечения
kdl_init = 0.87 #Соотношение между длительными и кратковременными
mu_init = 0.8 #Коэффициент приведения длины
tau_init = 150 #Время пожара
epsb2 = 3.5*10**(-3) #Предельные деформации бетона
Es = 2*10**5 #Модуль упругости стали
#Строка с армированием
Reinf_string_list = 's200d6, s200d8, s200d12, s200d16, s200d20, s200d25'
#Строка с длинами
Len_string_list = '1000, 2500, 3000, 3500, 4000, 4500, 5000, 5500'
a_init = 50
#название страницы
st.set_page_config(page_title='Расчет внецентренно сжатого железобетонного элемента')
#__________Ипорт классов бетона и арматуры__________
concrete_list = []
for k, v in Concretes.items():
    concrete_list.append(k)
reinf_list = []
for k, v in Steels.items():
    reinf_list.append(k)

st.title('Расчет внецентренно сжатого железобетонного элемента')

st.write('''Ниже выполняется расчет внецентренно сжатого железобетонного прямоугольного сечения,
шириной $b=100$см и высотой $h$, указанной пользователем. Предполагается, что величина статического
эксцентриситета $M/N$ не превышает случайный $e_a$. Результаты приводятся
для списка армирования и длин, указанных пользователем.
Армирование сечения - симметричное с привязкой $a$.''')


st.subheader('Основные параметры расчета')
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75])
with col1: h = st.number_input(label="$h$, см", step=5, format="%i", value=h_init, min_value=15, max_value=80)
b = b_init
with col2: a = st.number_input(label="$a$, мм", step=5, format="%i", value=a_init, min_value=15, max_value=100)
h0 = h - a
with col3: concrete = st.selectbox('Бетон', concrete_list, index=4)
cur_concrete = Concretes[concrete]
with col4: reinf = st.selectbox('Арматура', reinf_list, index=0)
cur_reinf = Steels[reinf]
with col5: coeff_b = st.number_input('$\gamma_b$', step=0.01, format="%.2f", value=coeff_b_init, min_value=0.099, max_value=1.0)
with col6: mu = st.number_input('$\mu$', step=0.1, format="%.2f", value=mu_init, min_value=0.099, max_value=5.0)
with col7: tau = st.number_input(label='$t$, мин', step=10, format="%i", value=tau_init, min_value=30, max_value=240)
with col8: kdl = st.number_input(label="Длит./Крат.", step=0.01, format="%.2f", value=kdl_init, min_value=0.099, max_value=1.0)

Rbc = cur_concrete['Rb']*coeff_b
Rbn = cur_concrete['Rbn']*coeff_b
Eb0 = cur_concrete['Eb0']
Rs = cur_reinf['Rs']
Rsn = cur_reinf['Rsn']
Rsc = cur_reinf['Rsck']
epssel = Rs/Es
xiR = 0.8 / (1 + epssel / epsb2)

st.subheader('Варианты расчета')
Reinf_string = st.text_input('Варианты армирования у ОДНОЙ грани. Разделители: запятая или пробел', value = Reinf_string_list)
Reinf_string_temp = re.split(';|,| ', Reinf_string)
Reinf_string_list = []
for i in Reinf_string_temp:
    try:
        if i!='':
            Reinf_string_list.append(i)
    except: pass
Reinf_string_list = sorted(Reinf_string_list, key=lambda x: calc_string_area(x), reverse=True)
Reinf_data = []
for i in Reinf_string_list:
    Reinf_data.append({'string': i, 'area_1': round(calc_string_area(i),3),
                       'area_2': round(2*calc_string_area(i),3),
                       'mu': round(2*calc_string_area(i)/(h*b-2*calc_string_area(i))*100,3)})
Reinf_data = pd.DataFrame(Reinf_data)
##st.data_editor(Reinf_data)

Len_string = st.text_input('Рассматриваемые длины в мм. Разделители: запятая или пробел', value = Len_string_list)
Len_data_temp = re.split(';|,| ', Len_string)
Len_data = []
for i in Len_data_temp:
    try: Len_data.append(int(i))
    except: pass
Len_data = [int(i) for i in Len_data]
Len_data = sorted(Len_data, reverse=False)

rez = []
for i in range(len(Reinf_data)):
    cur_As = Reinf_data['area_1'][i]
    rez_row = []
    rez_row.append(str(Reinf_data['mu'][i])+'%')
    rez_row.append(Reinf_data['area_1'][i])
    rez_row.append(Reinf_data['string'][i])
    for j in range(len(Len_data)):
        cur_L = Len_data[j]
        tmp = solve_N(b, h, cur_L/10, a/10,
                    mu, kdl, cur_As, cur_As,
                    Eb0/10, Es/10, xiR, Rs/10, Rsc/10, Rbc/10)[2]
        rez_row.append(str(-math.floor(tmp/10)*10))
    rez.append(rez_row)

col_conf = {'0': st.column_config.TextColumn(label='%', disabled=True, help='Процент армирования (суммарный у двух граней)'),
            '1': st.column_config.TextColumn(label='см2', disabled=True, help='Площадь арматуры у ОДНОЙ грани'),
            '2': st.column_config.TextColumn(label='As', disabled=True, help='Текстовое описание армирования у ОДНОЙ грани')}
for i in range(len(Len_data)):
    col_conf.update({str(i+3): st.column_config.TextColumn(
        label='L='+str(Len_data[i]), disabled=True, help='Эксцентриситет e=' + str(round(max(1.0, h/30, Len_data[i]/10/600),2))+'см')})


with st.expander('Результаты расчета по прочности'):
    st.subheader('Допустимое вертикальное напряжение, тс/м$^2$')
    st.dataframe(pd.DataFrame(rez), hide_index=True, column_config=col_conf)

with st.expander('Справочная информация по огнестойкости'):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('Коэффициент к прочности бетона, &#947;$_{bt}$')
        fig = plot_gammabt()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write('Коэффициент к модулю бетона, &#946;$_{b}$')
        fig = plot_betabt()
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns([1, 1])
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('Коэффициент к прочности арматуры, &#947;$_{st}$')
        fig = plot_gammast()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write('Коэффициент к модулю арматуры, &#946;$_{s}$')
        fig = plot_betast()
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns([1, 1])
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('КЛТР бетона, 1/C$^{\circ}\cdot$10$^6$')
        fig =  plot_alphab()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write('КЛТР арматуры, 1/C$^{\circ}\cdot$10$^6$')
        fig = plot_alphas()
        plotconf = dict({'staticPlot':True})
        st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('Глубина прогрева бетона, мм')
        fig = plot_lpr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write('Глубина прогрева до T$_{cr}$=500C$^{\circ}$, мм')
        fig = plot_at()
        plotconf = dict({'staticPlot':True})
        st.plotly_chart(fig, use_container_width=True)

    st.write('Распределение температуры T по толщине сечения, C$^{\circ}$')
    fig = plot_T(h, tau)
    plotconf = dict({'staticPlot':True})
    st.plotly_chart(fig, use_container_width=True)
    st.write('Температура со стороны нагрева: ' + str(round(Ta(a/1000, tau, 28/1000))) + '   ' + str(round(Tb(a/1000, tau))))

##st.write(str(solve_NF(b, h, 300, a/10, mu, kdl, 15, 15, Eb0/10, Es/10, xiR, Rsn/10, Rsc/10, Rbn/10, tau, 12/10)))

rez1 = []
for i in range(len(Reinf_data)):
    cur_As1 = Reinf_data['area_1'][i]
    rez1_row = []
    rez1_row.append(str(Reinf_data['mu'][i])+'%')
    rez1_row.append(Reinf_data['area_1'][i])
    rez1_row.append(Reinf_data['string'][i])
    for j in range(len(Len_data)):
        cur_L1 = Len_data[j]
        tmp1 = solve_NF(b, h, cur_L1/10, a/10, mu, kdl, cur_As1, cur_As1, Eb0/10, Es/10, xiR, Rsn/10, Rsc/10, Rbn/10, tau, d/10)[2]
        rez1_row.append(str(-math.floor(tmp1/10)*10))
    rez1.append(rez1_row)

col_conf = {'0': st.column_config.TextColumn(label='%', disabled=True, help='Процент армирования (суммарный у двух граней)'),
            '1': st.column_config.TextColumn(label='см2', disabled=True, help='Площадь арматуры у ОДНОЙ грани'),
            '2': st.column_config.TextColumn(label='As', disabled=True, help='Текстовое описание армирования у ОДНОЙ грани')}

for i in range(len(Len_data)):
    col_conf.update({str(i+3): st.column_config.TextColumn(
        label='L='+str(Len_data[i]), disabled=True, help='Эксцентриситет e=' + str(round(max(1.0, h/30, Len_data[i]/10/600),2))+'см')})
    
with st.expander('Результаты расчета огнестойкости'):
    st.subheader('Допустимое вертикальное напряжение, тс/м$^2$')
    st.dataframe(pd.DataFrame(rez1), hide_index=True, column_config=col_conf)
