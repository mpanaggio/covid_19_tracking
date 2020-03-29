import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress
np.set_printoptions(suppress=True)
from ipywidgets import IntSlider,FloatSlider,interact,Checkbox,GridBox,Layout,interactive_output,FloatText,Dropdown,Select,IntText,SelectMultiple
from ipywidgets import HBox,Label,SelectionSlider,Box,FloatRangeSlider
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


df=pd.read_csv('state_by_state_data.csv',parse_dates=True)
df=df[df['Country/Region']=='US']
colors='#e6194B, #3cb44b, #ffe119, #4363d8, #f58231, #911eb4, #42d4f4, #f032e6, #bfef45, #fabebe, #469990, #e6beff, #9A6324, #fffac8, #800000, #aaffc3, #808000, #ffd8b1, #000075, #a9a9a9'.split(', ')

def create_selection_box():
    states=df['Province/State'].unique().tolist()
    peaks=[]
    for s in states:
        try:
            tmp=df[df['Province/State']==s].sum().drop(['Country/Region','Lat','Long','Province/State']).max()
        except:
            try:
                tmp=df[df['Province/State']==c].sum().drop(['Lat','Long']).max()
            except:
                tmp=0
        peaks.append(tmp)
    tmp_zip=sorted(list(zip(peaks,states)),reverse=True)
    states=[t[1] for t in tmp_zip]
    #countries.sort()
    states=['All']+states
    style = {'description_width': 'initial','width': 100}
    select=SelectMultiple(
        options=states,
        value=['NY'],
        style=style,
        disabled=False
    )
    select=HBox([Label('Select a state (or multiple states):'), select])
    display(select)
    return select,states

def compute_data_props(states,select):
    state=[states[k] for k in select.children[1].index]
    df_dict={}
    maxval=0
    for c in state:
        if c=='All':
            df_dict[c]=df.sum().drop(['Lat','Long','Country/Region','Province/State'])
        else:
            try:
                df_dict[c]=df[df['Province/State']==c].sum().drop(['Country/Region','Lat','Long','Province/State'])
            except:
                df_dict[c]=df[df['Province/State']==c].sum().drop(['Lat','Long'])
                raise ValueError('Something went wrong!')
        if df_dict[c].max()>maxval:
            maxval=df_dict[c].max()
    last_month,last_day,x=df.columns[-1].split('/')
    last_month=int(last_month)
    last_day=int(last_day)
    return last_month,last_day,maxval,state,df_dict
def make_interactive_plot(def_plotlog=False,
                          start_month_plot=3,start_day_plot=4,
                          end_month_plot=3,end_day_plot=24,
                          def_start_month_fit=3,def_start_day_fit=4, 
                          def_end_month_fit=3,def_end_day_fit=24,
                          def_month_pred=3,def_day_pred=24,maxval=70000,country='NY',df_dict={}):
    
    style = {'description_width': 'initial'}
    xrange=FloatRangeSlider(value=[-0.1,1.1],min=-0.2,max=1.5,step=0.01,description='x range',readout=False)
    ymax=FloatSlider(value=maxval*1.25,min=0,max=maxval*1.25,step=100,style=style,description='y max')
    start_month_fit=IntText(value=def_start_month_fit, description='Start month for exponential fit',style=style,layout=Layout(width='70%', height='30px'))
    start_day_fit=IntText(value=def_start_day_fit, description='Start day for exponential fit ',style=style,layout=Layout(width='70%', height='30px'))

    end_month_fit=IntText(value=def_end_month_fit, description='End month for exponential fit',style=style,layout=Layout(width='70%', height='30px'))
    end_day_fit=IntText(value=def_end_day_fit, description='End day for exponential fit',style=style,layout=Layout(width='70%', height='30px'))
    
    month_pred=IntText(value=def_month_pred, description='Month for prediction',style=style,layout=Layout(width='70%', height='30px'))
    day_pred=IntText(value=def_day_pred, description='Day for prediction',style=style,layout=Layout(width='70%', height='30px'))
    plotlog=Checkbox(value=def_plotlog,description='logarithmic y-axis?')
    fig, ax = plt.subplots(figsize=(10,6))
    def make_plot(c,color,plotlog=def_plotlog,
                  start_month_fit=def_start_month_fit,start_day_fit=def_start_day_fit, 
                  end_month_fit=def_end_month_fit,end_day_fit=def_end_day_fit,
                  month_pred=def_month_pred,day_pred=def_day_pred):
        tmp=df_dict[c]
        time=pd.to_datetime(tmp.index).values
        counts=tmp.values.astype(np.float64)
        start_ind=np.argwhere(time==np.datetime64(datetime.date(2020, start_month_plot, start_day_plot)))[0][0]
        try:
            end_ind=np.argwhere(time==np.datetime64(datetime.date(2020, end_month_plot, end_day_plot)))[0][0]
            time=time[start_ind:end_ind+1]
            counts=counts[start_ind:end_ind+1]
        except:
            time=time[start_ind:]
            counts=counts[start_ind:]


        start_ind_fit=np.argwhere(time==np.datetime64(datetime.date(2020, start_month_fit, start_day_fit)))[0][0]
        try:
            end_ind_fit=np.argwhere(time==np.datetime64(datetime.date(2020, end_month_fit, end_day_fit)))[0][0]
        except Exception as e:
            end_ind_fit=-1
        time_start_fit=time[start_ind_fit]
        time_end_fit=time[end_ind_fit]
        time_int=time.astype(np.int64)
        ref=time_int[0]
        time_int-=ref
        ref2=time_int.max()
        time_int=time_int/ref2

        if plotlog:

            lpts,=plt.semilogy(time,counts,'.',color=color,label=c,linewidth=2)
        else:
            lpts,=plt.plot(time,counts,'.',color=color,label=c,linewidth=2)
        slope,intercept=np.polyfit(time_int[start_ind_fit:end_ind_fit],np.log10(counts[start_ind_fit:end_ind_fit]+1),deg=1)
        y=10**(slope*time_int+intercept)
        lcurve,=plt.plot(time,y,colors[counter],linewidth=2)

        prediction_time=np.datetime64(datetime.datetime(2020, month_pred, day_pred))
        prediction_time_int=(prediction_time.astype(np.int64)*1000-ref)/ref2
        #print("Predicted # of cases in {} at time {}: {}".format(c,datetime.date(2020, month_pred, day_pred),int(10**(slope*prediction_time_int+intercept))) )
        return lpts,lcurve,time_start_fit,time_end_fit
    
    lines={}
    counter=0
    for c in country:
        lpts,lcurve,time_start_fit,time_end_fit=make_plot(c,color=colors[counter])
        counter+=1
        lines[c]={'pts': lpts,'curve':lcurve}
    lstart=plt.axvline(time_start_fit)
    lend=plt.axvline(time_end_fit)
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')
        
    ui=GridBox(children=[xrange,ymax,
               start_month_fit,start_day_fit, end_month_fit,end_day_fit,
               month_pred,day_pred,plotlog
                     ],
        layout=Layout(
            width='100%',
            grid_template_rows='auto auto',
            grid_template_columns='45% 45%')
       )
    def update(xrange,ymax,
               start_month_fit,start_day_fit, end_month_fit,end_day_fit,
               month_pred,day_pred,plotlog):
        for c in country:
            tmp=df_dict[c]
            time=pd.to_datetime(tmp.index).values
            counts=tmp.values.astype(np.float64)

            try:
                start_ind=np.argwhere(time==np.datetime64(datetime.date(2020, start_month_plot, start_day_plot)))[0][0]
            except:
                print("Invalid start date for plot. Using first point.")
                start_ind=0

            try:
                end_ind=np.argwhere(time==np.datetime64(datetime.date(2020, end_month_plot, end_day_plot)))[0][0]
                time=time[start_ind:end_ind+1]
                counts=counts[start_ind:end_ind+1]
            except:
                print("Invalid end date for plot. Using last point.")
                time=time[start_ind:]
                counts=counts[start_ind:]

            try:
                start_ind_fit=np.argwhere(time==np.datetime64(datetime.date(2020, start_month_fit, start_day_fit)))[0][0]
            except:
                print("Invalid start date for fit. Using first point.")
                start_ind_fit=0
            try:
                end_ind_fit=np.argwhere(time==np.datetime64(datetime.date(2020, end_month_fit, end_day_fit)))[0][0]
            except: 
                print("Invalid end date for fit. Using last point.")
                end_ind_fit=-1
            lstart.set_xdata(time[start_ind_fit])
            lend.set_xdata(time[end_ind_fit])
            time_int=time.astype(np.int64)
            ref=time_int[0]
            time_int-=ref
            ref2=time_int.max()
            time_int=time_int/ref2
            lines[c]['pts'].set_xdata(time)
            lines[c]['pts'].set_ydata(counts)
            slope,intercept=np.polyfit(time_int[start_ind_fit:end_ind_fit],np.log10(counts[start_ind_fit:end_ind_fit]+1),deg=1)
            y=10**(slope*time_int+intercept)
            lines[c]['curve'].set_ydata(time)
            lines[c]['curve'].set_ydata(y)
            if plotlog:
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')
            try:
                prediction_time=np.datetime64(datetime.datetime(2020, month_pred, day_pred))
                prediction_time_int=(prediction_time.astype(np.int64)*1000-ref)/ref2
                print("Predicted # of cases in {} at time {}: {}".format(c,datetime.date(2020, month_pred, day_pred),int(10**(slope*prediction_time_int+intercept))) )
            except:
                print('Invalid date for prediction.')
        plt.xlim(xrange[0]*(time[-1]-time[0])+time[0],xrange[1]*(time[-1]-time[0])+time[0])
        plt.ylim(-0.1*ymax,ymax)
        fig.canvas.draw_idle()
           

    output=interactive_output(update,  {'xrange': xrange,
                                        'ymax': ymax,
                                        'start_month_fit': start_month_fit,
                                        'start_day_fit': start_day_fit,
                                        'end_month_fit': end_month_fit,
                                        'end_day_fit': end_day_fit,
                                        'month_pred': month_pred,
                                        'day_pred': day_pred,
                                        'plotlog': plotlog
                                       }
                                 )  
    return output,ui, lines,lstart,lend