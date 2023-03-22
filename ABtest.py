#!/usr/bin/env python
# coding: utf-8

# ### Проект по А/B-тестированию

# ### Постановка задачи
# Ваша задача — провести оценку результатов A/B-теста. В вашем распоряжении есть датасет с действиями пользователей, техническое задание и несколько вспомогательных датасетов.
# * Оцените корректность проведения теста
# * Проанализируйте результаты теста
# 
# Чтобы оценить корректность проведения теста, проверьте:
# * пересечение тестовой аудитории с конкурирующим тестом,
# * совпадение теста и маркетинговых событий, другие проблемы временных границ теста.

# ### Техническое задание
# * Название теста: recommender_system_test;
# * Группы: А (контрольная), B (новая платёжная воронка);
# * Дата запуска: 2020-12-07;
# * Дата остановки набора новых пользователей: 2020-12-21;
# * Дата остановки: 2021-01-04;
# * Аудитория: 15% новых пользователей из региона EU;
# * Назначение теста: тестирование изменений, связанных с внедрением улучшенной рекомендательной системы;
# * Ожидаемое количество участников теста: 6000.
# * Ожидаемый эффект: за 14 дней с момента регистрации в системе пользователи покажут улучшение каждой метрики не менее, чем на 10%:
#     * конверсии в просмотр карточек товаров — событие product_page
#     * просмотры корзины — product_card
#     * покупки — purchase.
# 
# Загрузите данные теста, проверьте корректность его проведения и проанализируйте полученные результаты.

# ### Данные
# ab_project_marketing_events.csv
# final_ab_new_users.csv
# final_ab_events.csv
# final_ab_participants.csv
# 
# 
# /datasets/ab_project_marketing_events.csv — календарь маркетинговых событий на 2020 год;
# 
# Структура файла:
# 
# * name — название маркетингового события;
# * regions — регионы, в которых будет проводиться рекламная кампания;
# * start_dt — дата начала кампании;
# * finish_dt — дата завершения кампании.
# 
# 
# /datasets/final_ab_new_users.csv — все пользователи, зарегистрировавшиеся в интернет-магазине в период с 7 по 21 декабря 2020 года;
# 
# Структура файла:
# * user_id — идентификатор пользователя;
# * first_date — дата регистрации;
# * region — регион пользователя;
# * device — устройство, с которого происходила регистрация.
# 
# 
# /datasets/final_ab_events.csv — все события новых пользователей в период с 7 декабря 2020 по 4 января 2021 года;
# 
# Структура файла:
# * user_id — идентификатор пользователя;
# * event_dt — дата и время события;
# * event_name — тип события;
# * details — дополнительные данные о событии. Например, для покупок, purchase, в этом поле хранится стоимость покупки в долларах.
# 
# 
# /datasets/final_ab_participants.csv — таблица участников тестов.
# 
# Структура файла:
# * user_id — идентификатор пользователя;
# * ab_test — название теста;
# * group — группа пользователя.

# Как сделать задание?
# * Опишите цели исследования
# * Исследуйте данные
#     * Требуется ли преобразование типов?
#     * Присутствуют ли пропущенные значения и дубликаты? Если да, то какова их природа?
# * Проведите исследовательский анализ данных
#     * Исследуйте конверсию в воронке на разных этапах?
#     * Обладают ли выборки одинаковыми распределениями количества событий на пользователя?
#     * Присутствуют ли в выборках одни и те же пользователи?
#     * Как число событий распределено по дням?
#     * Подумайте, есть ли какие-то нюансы данных, которые нужно учесть, прежде чем приступать к A/B-тестированию?
# * Проведите оценку результатов A/B-тестирования
#     * Что можно сказать про результаты A/B-тестирования?
#     * Проверьте статистическую разницу долей z-критерием
# * Опишите выводы по этапу исследовательского анализа данных и по проведённой оценке результатов A/B-тестирования

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)})

import scipy.stats as stats
from scipy import stats as st

import math as mth

import numpy as np

import pandas as pdm
from datetime import datetime,timedelta

from pathlib import Path
import matplotlib.dates as mdates

import math
import cmath

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# ### Исследуйте данные

# In[2]:


ab_project_marketing_events = pd.read_csv('/datasets/ab_project_marketing_events.csv', sep=',')
final_ab_new_users = pd.read_csv('/datasets/final_ab_new_users.csv', sep=',')
final_ab_events = pd.read_csv('/datasets/final_ab_events.csv', sep=',')
final_ab_participants = pd.read_csv('/datasets/final_ab_participants.csv', sep=',')


# In[3]:


ab_project_marketing_events.info()


# In[4]:


final_ab_new_users.info()


# In[5]:


final_ab_events.info()


# In[6]:


final_ab_events['details'].value_counts()


# In[7]:



final_ab_events['details'] = final_ab_events['details'].fillna(0)


# In[8]:


final_ab_events.info()


# In[9]:


final_ab_participants.info()


# In[10]:


ab_project_marketing_events.duplicated().sum()


# In[11]:


final_ab_new_users.duplicated().sum()


# In[12]:


final_ab_events.duplicated().sum()


# In[13]:


final_ab_participants.duplicated().sum()


# In[14]:


final_ab_events['event_dt'] = final_ab_events['event_dt'].astype('datetime64')


# In[15]:


final_ab_new_users['first_date'] = final_ab_new_users['first_date'].astype('datetime64')


# In[16]:


ab_project_marketing_events['start_dt'] = ab_project_marketing_events['start_dt'].astype('datetime64')
ab_project_marketing_events['finish_dt'] = ab_project_marketing_events['finish_dt'].astype('datetime64')


# In[17]:


final_ab_events['event_name'].unique()


# пропуски говорят о том, что событие было бесплатным

# ### Проведите исследовательский анализ данных

# #### Исследуйте конверсию в воронке на разных этапах?

# In[18]:


final_ab_new_users = pd.merge(final_ab_new_users, final_ab_participants, how = 'left')
final_ab_new_users = final_ab_new_users.dropna()
final_ab_new_users


# In[19]:


final_ab_events = pd.merge(final_ab_events, final_ab_new_users, how = 'left')
final_ab_events = final_ab_events.dropna()
final_ab_events


# In[20]:


final_ab_new_users_grouped = final_ab_new_users.groupby(['ab_test','group'], as_index=False).agg({'user_id':'count'}).sort_values(['ab_test','group'])
final_ab_new_users_grouped


# * Название теста: recommender_system_test;
# * Группы: А (контрольная), B (новая платёжная воронка);
# * Дата запуска: 2020-12-07;
# * Дата остановки набора новых пользователей: 2020-12-21;
# * Дата остановки: 2021-01-04;
# * Аудитория: 15% новых пользователей из региона EU;

# In[21]:


final_ab_events_copy = final_ab_events.copy()


# In[22]:


filtered_users = final_ab_events.query("ab_test != 'recommender_system_test'")['user_id']


# In[23]:


final_ab_events = final_ab_events.query("region == 'EU' and event_dt > '2020-12-07' and event_dt < '2020-12-21'")
final_ab_events = final_ab_events.query('user_id not in @filtered_users')


# In[24]:


other_test_users = final_ab_events_copy.query("ab_test != 'recommender_system_test'")['user_id']
final_ab_events.query('user_id in @other_test_users')


# In[25]:


final_ab_events.head()


# In[26]:


final_ab_events['session_week'] = final_ab_events['event_dt'].dt.week
final_ab_events['session_date'] = final_ab_events['event_dt'].dt.date


# In[27]:


final_ab_events['session_date'] = final_ab_events['session_date'].astype('datetime64')


# In[28]:


dau_total = final_ab_events.groupby('session_date').agg({'user_id': 'nunique'}).mean()
wau_total = final_ab_events.groupby(['session_week']).agg({'user_id': 'nunique'}).mean()


# In[29]:


dau_total_gr = final_ab_events.groupby('session_date').agg({'user_id': 'nunique'})
wau_total_gr = final_ab_events.groupby(['session_week']).agg({'user_id': 'nunique'})


# In[30]:


ax_dau = dau_total_gr.plot()
ax_dau.set_title('Зависимость посещения по дням')
ax_dau.set_xlabel('Дата')
ax_dau.set_ylabel('Посещения')


# In[31]:


ax_wau = wau_total_gr.plot()
ax_wau.set_title('Зависимость посещения по неделям')
ax_wau.set_xlabel('Дата')
ax_wau.set_ylabel('Посещения')


# In[32]:


final_ab_events.nunique()


# повторяющиеся значения по user_id есть

# In[33]:


event = final_ab_events.groupby('event_dt').agg({'event_name':'count'})


# In[34]:


ax = event.plot()
ax.set_title('Как число событий распределено по дням')
ax.set_xlabel('Дата')
ax.set_ylabel('События')


# In[35]:


final_ab_events.info()


# In[36]:


final_ab_events_count = (final_ab_events
                .groupby(['event_name'])['user_id'].nunique()
                .reset_index()
                .rename(columns={'user_id':'counts'})
                        )
final_ab_events_count


# In[37]:


df = pd.merge(final_ab_new_users, final_ab_events,on='user_id', how = 'left')
df['details'].unique()


# In[38]:


final_ab_events_sum = (final_ab_events
                .groupby(['region'])['details'].sum()
                .reset_index()
                .rename(columns={'details':'sum_details'})
                        )
final_ab_events_sum


# In[39]:


fig = px.bar(final_ab_events_count, x='event_name', y='counts', color='event_name')
fig.update_xaxes(tickangle=45)
fig.show()


# In[40]:


fig = px.bar(final_ab_events_sum, x='region', y='sum_details', color='region')
fig.update_xaxes(tickangle=45)
fig.show()


# In[41]:


event = final_ab_events.groupby('user_id').agg({'details':'count'})
event


# In[42]:


matplotlib.style.use('ggplot')

s = pd.Series(event['details'])

s.plot.kde()


# In[43]:


final_ab_events


# покупок больше чем переходов на product_card

# In[44]:


ab_project_marketing_events = ab_project_marketing_events.rename(columns={'regions': 'region'})


# In[45]:


df = pd.merge(final_ab_new_users, ab_project_marketing_events,on='region', how = 'left')


# In[46]:


final_ab_new_users


# In[47]:


df = pd.merge(final_ab_new_users, ab_project_marketing_events,on='region', how = 'left')
df


# In[48]:


df = pd.merge(df,final_ab_events,on=['user_id'], how = 'left')
df


# #### Не все маркетинговые события подходят к нашей выборке

# * Christmas&New Year Promo, CIS New Year Gift Lottery проходят в период теста 
# * Christmas&New Year Promo точно повлиял на результаты, что нельзя сказать о New Year Gift Lottery так как мы не располагаем данными по значительному времени данной акции 
# 

# ### Проведите оценку результатов A/B-тестирования

# * Н0 - группы, которые используют Mac и PC в европе различаются
# * Н1 - группы статистически одинаковы

# In[49]:


final_ab_events['group'] = [1 if x == 'A' else 2 for x in final_ab_events['group']]


# In[50]:


final_ab_events_group = final_ab_events.groupby(['ab_test','group'], as_index=False).agg({'user_id':'count'}).sort_values(['ab_test','group'])
final_ab_events_group


# In[51]:


test = final_ab_events.pivot_table(index='event_name', columns='group',values='user_id',aggfunc='nunique')
test


# In[52]:


final_ab_events_copy = final_ab_events_copy.query("event_dt > '2020-12-21' and event_dt < '2020-01-04'")


# In[53]:


test1 = final_ab_events_copy.pivot_table(index='event_name', columns='group',values='user_id',aggfunc='nunique')
test1


# In[54]:


test.sum()


# In[55]:


people = final_ab_events.groupby('group')['user_id'].nunique()
users = people.to_frame().reset_index()
users = users.set_index(users.columns[0])
users


# * H0 - между группами А и Б нет различимой разницы
# * H1 - выборки отличаются между собой 

# In[56]:


def z_test(groupA, groupB, event, alpha): 
    p1_ev = test.loc[event, groupA]
    p2_ev = test.loc[event, groupB] 
    p1_us = users.loc[groupA, 'user_id'] 
    p2_us = users.loc[groupB, 'user_id'] 
    p1 = p1_ev / p1_us 
    p2 = p2_ev / p2_us 
    difference = p1 - p2
    p_combined = (p1_ev + p2_ev) / (p1_us + p2_us) 
    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1 / p1_us + 1 / p2_us))
    distr = st.norm(0, 1)
    p_value = ((1 - distr.cdf(abs(z_value))) * 2)
    print('Проверка для  {} и {}, событие: {}, p-значение: {p_value:.2f}'.format(groupA, groupB, event, p_value=p_value))
    if (p_value < alpha):
        print("Отвергаем нулевую гипотезу")
    else:
        print("Не получилось отвергнуть нулевую гипотезу")


# In[57]:


for event in test.index:
    z_test(1, 2, event, 0.0125)
    print()


# alpha/4 = 0.0125

# Использовали поправку Бонферонни для множественных тестов 

# ## Выводы

# * Самые большие прибыли в EU регионе, второй в рейтинге - N. America
# * униркальных покупателей 19,5к - это на 200 человек больше чем предыдущее число в воронке, связано скорее всего с количеством товара или покупка в составе набора 
# * 2 пика по датам: если пик 23ого числа можно связать с Рождеством, то по пику 14ого числа у меня пока нет идей
# * к концу месяца посещение падает - пользователи готовяться к Новому Году
# * Гипотезы подтвердились 

# * группа А выглядит лучше

# ### Вывод:

# * пик по датам с 13 по 14 число - связано скорее с тем, что в этот период наступает рождество в Европе, далее прирост можно объяснить наступлением нового года 
# * отфильтровали данные: 
#     * по дате запуска: 2020-12-07  и дата остановки набора новых пользователей: 2020-12-21
#     * пользователей с даты остановки набора новых пользователей: 2020-12-21 и дате остановки: 2021-01-04 в тесте нет 
#     * по региону EU
#     * по рекомендуемому тесту recommender_system_test
#     
# * попадают 2 маркетинговых события, одно из которых влияет на наше исследование: Christmas&New Year Promo и CIS New Year Gift Lottery
#     * CIS New Year Gift Lottery: не влияет, т.к. как отмечено выше, данных в тесте нет по датам данного события 
# 
# 
# * Мы проверили гипотезы: 
#     * Проверка для  1 и 2, событие: login, p-значение: 0.10
#     тест показал что пользователи, которые залогинились отличаются по группам 
# 
#     * Проверка для  1 и 2, событие: product_cart, p-значение: 0.26
#     тест показал что просмотры карточек товаров отличаются по группам 
# 
#     * Проверка для  1 и 2, событие: product_page, p-значение: 0.00
#     тест показал что просмотры корзины одинаковые по группам 
# 
#     * Проверка для  1 и 2, событие: purchase, p-значение: 0.15
#     тест показал что покупки отличаются по группам 
# 
# ### Рекомендации:
# * Определить различие между тестируемыми группами А и Б
# * Оставить приоритет за группой А, т.к. она "выигрывает" по приоритетным параматрам 
# * так как маркетинговые события сильно влияют на исследования в будущем надо проводить тесты по возможности исключая влияние маркетинговых событий
# 
