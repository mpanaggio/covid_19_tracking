import pandas as pd
import numpy as np
import seaborn as sns
sns.set(font_scale=2)
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
from ipywidgets import interact,Layout,interactive_output,SelectMultiple,HBox,Label

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('merged_us_data_by_state_with_mobility_and_policy.csv',index_col=0,parse_dates=[1]).reset_index(drop=True)

df=df.rename(columns={'retail_and_recreation_percent_change_from_baseline': 'retail and recreation',
                      'grocery_and_pharmacy_percent_change_from_baseline': 'grocery and pharmacy',
                      'parks_percent_change_from_baseline': 'parks',
                      'transit_stations_percent_change_from_baseline': 'transit',
                      'workplaces_percent_change_from_baseline': 'workplaces',
                      'residential_percent_change_from_baseline': 'residential'})
df=df[~df.state.isin(['PR','AS','MP','VI','GU'])]

states=df.state.unique()
state_tuples_by_infected=[]
state_tuples_by_deaths=[]
for state in states:
    total_infected=df[df.state==state]['positive'].max()
    total_deaths=df[df.state==state]['death'].max()
    state_tuples_by_infected.append((total_infected,state))
    state_tuples_by_deaths.append((total_deaths,state))
state_tuples_by_infected.sort(reverse=True)
state_tuples_by_deaths.sort(reverse=True)

states_by_infected=[c[1] for c in state_tuples_by_infected]
states_by_deaths=[c[1] for c in state_tuples_by_deaths]
national_df=df.groupby('date').sum()

def align_dates(df,thr=100):
    df=df.sort_values(by='date').reset_index()
    ind_thr=(df['positive']>=thr).idxmax()
    ref_date=df.loc[ind_thr,'date']
    df['date']=(df['date']-ref_date)
    df['date']=df['date'].apply(lambda x: x.days)
    return df

first=True
for state in states:
    if first:
        df_aligned=align_dates(df[df.state==state])
        first=False
    else:
        df_aligned=pd.concat([df_aligned,align_dates(df[df.state==state])],axis=0)
df_aligned.reset_index(inplace=True)

def plot_vs_events(state,col,aligned=False,logy=False,logx=False,events=False,ylabel='',mvavg=1):
    if aligned:
        mydf=df_aligned[df.state==state]
    else:
        mydf=df[df.state==state]
    mydf=mydf.reset_index(drop=True)

    plt.plot(mydf.date[mvavg-1:],moving_average(mydf[col],n=mvavg),label=col)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if events:
        if mydf['pandemic declared'].idxmax()>0:
            plt.axvline(mydf.loc[mydf['pandemic declared'].idxmax(),'date'],
                    label='pandemic declared',linestyle='--',color='g')
        if mydf['school canceled'].idxmax()>0:
            plt.axvline(mydf.loc[mydf['school canceled'].idxmax(),'date'],
                    label='school canceled',linestyle='-.',color='y')
        if mydf['stay at home'].idxmax()>0:
            plt.axvline(mydf.loc[mydf['stay at home'].idxmax(),'date'],
                    label='stay at home',linestyle=':',color='r')
        #'stay at home', 'pandemic declared','school canceled'
    if aligned:
        plt.xlabel('days since 100th case')
    else:
        plt.ylabel('date')
        plt.xticks(rotation=90)
    if len(ylabel)>0:
        plt.ylabel(ylabel)
        
def plot_mobility(state,aligned,mvavg=1):
    plt.figure(figsize=(12,6))
    plt.title('Change in mobility for {}'.format(state))
    plot_vs_events(state,col='retail and recreation',aligned=aligned,mvavg=mvavg)
    plot_vs_events(state,col='grocery and pharmacy',aligned=aligned,mvavg=mvavg)
    #plot_vs_events(state,col='parks',aligned=True,mvavg=mvavg)
    plot_vs_events(state,col='transit',aligned=aligned,mvavg=mvavg)
    plot_vs_events(state,col='workplaces',aligned=aligned,mvavg=mvavg)
    plot_vs_events(state,col='residential',aligned=aligned,mvavg=mvavg,logy=False,
                   events=True,ylabel='percent change\n relative to baseline')
    plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', ncol=1)
    plt.show()
    
def make_widgets_mobility():
    style = {'description_width': 'initial','width': 100}
    select=SelectMultiple(
            options=states_by_infected,
            value=[states_by_infected[0]],
            style=style,
            disabled=False
        )
    select=HBox([Label('Select a state (or multiple states):'), select])
    display(select)
    return select

def fit_and_plot_test_results():
    tvec=np.array((national_df.index-national_df.index[0]).days)
    logistic = lambda x,a,b,c: c/(1+np.exp(-a*(x-b)))
    plt.figure(figsize=(12,6))
    popt, pcov = curve_fit(logistic, tvec,national_df['totalTestResultsIncrease'].values)
    logistic_parameters=popt
    plt.title('Nationwide test totals')
    plt.plot(tvec,national_df['totalTestResultsIncrease'],label='tests administered')
    R2=r2_score(y_true=national_df['totalTestResultsIncrease'].values,
               y_pred=logistic(tvec, *popt))
    plt.plot(tvec, logistic(tvec, *popt), 'r-', label='logistic fit ($R^2$={:.3f})'.format(R2))
    plt.plot(tvec,popt[2]+0*tvec,label='daily testing capacity \n estimate: {}'.format(int(np.round(popt[2]))),color='k',linestyle='--')
    plt.xlabel('days since {}/{}'.format(national_df.index[0].month,national_df.index[0].day))
    plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', ncol=1)
    plt.show()
    
def make_widgets_testing():
    from ipywidgets import interact,Layout,interactive_output,SelectMultiple,HBox,Label,IntSlider
    style = {'description_width': 'initial','width': 100}
    select=SelectMultiple(
            options=states_by_infected,
            value=[states_by_infected[0]],
            style=style,
            disabled=False
        )
    select=HBox([Label('Select a state (or multiple states):'), select])

    mov_avg=IntSlider(
        value=1,
        min=1,
        max=14,
        step=1,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    mov_avg=HBox([Label('Select a window size. Values larger than one use a moving average to smooth the curve:'), mov_avg])
    display(select)
    display(mov_avg)
    return select,mov_avg

def moving_average(a, n=3) :
    a=pd.Series(a).rolling(window=n).mean().iloc[n-1:].values
    return a

def plot_test_results(state,n):
    my_df=df[df.state==state]
    my_df=my_df.set_index('date')
    
    tvec=np.array((my_df.index-my_df.index[0]).days)
    yvec=my_df['totalTestResultsIncrease'].values
    plt.figure(figsize=(12,6))
    plt.title('Tests for {}'.format(state))
    plt.plot(moving_average(tvec,n=n),moving_average(yvec,n=n),label='tests administered')
    plt.xlabel('days since {}/{}'.format(my_df.index[0].month,national_df.index[0].day))
    #plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', ncol=1)
    plt.show()

def plot_positive_rate(window=1):
    fig,ax=plt.subplots(2,1,figsize=(12,8),sharex=True)
    ax[0].plot(national_df.index[window-1:],moving_average(national_df['positiveIncrease'],n=window)/moving_average(national_df['totalTestResultsIncrease'],n=window),label='% positive')
    ax[1].plot(national_df.index[window-1:],moving_average(national_df['totalTestResultsIncrease'],n=window),label='tests administered')
    ax[1].plot(national_df.index[window-1:],moving_average(national_df['positiveIncrease'],n=window),label='positive tests')
    ax[1].set_xlabel('date')
    ax[0].set_ylabel('% positive')
    #ax[1].set_ylabel('tests administered')
    ax[0].set_ylim(0,1.1)
    ax[1].legend()
    plt.xticks(rotation=90)
    plt.show()
    
def plot_positive_rate_by_state(state,window):
    my_df=df[df.state==state]
    my_df=my_df.set_index('date')
    
    tvec=np.array((my_df.index-my_df.index[0]).days)
    yvec=my_df['totalTestResultsIncrease'].values
    fig,ax=plt.subplots(2,1,figsize=(12,8),sharex=True)
    ax[0].set_title('Tests for {}'.format(state))
    ax[0].plot(my_df.index[window-1:],moving_average(my_df['positiveIncrease'],n=window)/moving_average(my_df['totalTestResultsIncrease'],n=window),label='% positive')
    ax[1].plot(my_df.index[window-1:],moving_average(my_df['totalTestResultsIncrease'],n=window),label='tests administered')
    ax[1].plot(my_df.index[window-1:],moving_average(my_df['positiveIncrease'],n=window),label='positive tests')
    ax[1].set_xlabel('date')
    ax[0].set_ylabel('% positive')
    #ax[1].set_ylabel('tests administered')
    ax[0].set_ylim(0,1.1)
    ax[1].legend()
    plt.xticks(rotation=90)
    plt.show()