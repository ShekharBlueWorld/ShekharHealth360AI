import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Shekhar Health360 AI',page_icon='💠',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#f5f8fc;color:#183149}.block-container{max-width:1320px;padding-top:1rem;padding-bottom:4rem}
h1,h2,h3{color:#172d44!important;letter-spacing:-.035em}h2{font-size:1.5rem!important}
.hero{background:linear-gradient(115deg,#102b4e 0%,#126e83 60%,#19a68b 100%);color:white;padding:28px 30px;border-radius:24px;box-shadow:0 12px 28px #17476920;margin-bottom:18px}
.hero h1{color:white!important;margin:0;font-size:2.2rem}.hero p{color:#d7f8f5;margin:8px 0 0}
div[data-testid="stMetric"]{background:white;border:1px solid #e0e8f1;border-radius:17px;padding:15px 18px;box-shadow:0 5px 18px #18324c0b;border-top:4px solid #0e9c8a}
div[data-testid="stMetricLabel"]{color:#52667a}div[data-testid="stMetricValue"]{color:#183149}
.insight{background:white;border:1px solid #e1e9f1;border-left:6px solid #148e82;border-radius:17px;padding:18px 20px;margin:10px 0;box-shadow:0 5px 18px #193a5109}
.insight h4{margin:0 0 8px;color:#1b3047;font-size:1.05rem}.insight p{margin:6px 0;color:#4a6174;font-size:.93rem}.insight b{color:#163d50}
.insight.amber{border-left-color:#e9a33a;background:#fffdf8}.insight.blue{border-left-color:#368cc9;background:#f8fcff}.insight.green{border-left-color:#19a17e;background:#f7fefa}.insight.violet{border-left-color:#8d76c7;background:#fcfaff}
.scorecard{background:white;border:1px solid #e3eaf2;border-radius:20px;padding:18px 19px;box-shadow:0 7px 22px #173b5510;min-height:155px}.scorecard .eyebrow{font-size:.78rem;font-weight:700;color:#60768a}.scorecard .big{font-size:2rem;font-weight:800;color:#172f47;letter-spacing:-.04em;margin:7px 0 2px}.scorecard .hint{font-size:.77rem;color:#5b7082}.scorecard .stripe{height:5px;border-radius:10px;margin-top:14px}.section-note{font-size:.87rem;color:#64798b;margin:-8px 0 12px}.tag{font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;font-weight:800;color:#627e94;margin-bottom:7px}
.smallprint{font-size:.8rem;color:#637789}
div[data-testid="stTabs"] button{font-weight:700!important;font-size:.94rem!important}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#087e79!important;border-bottom:3px solid #10a998!important}
div[data-testid="stExpander"]{border-radius:15px!important;border:1px solid #dce9ef!important;background:#ffffff}
div[data-testid="stSelectbox"]>div{border-radius:12px}
.stButton>button{border-radius:12px;font-weight:700}
.metric-strip{background:linear-gradient(125deg,#e8f8f4,#f0f5ff);padding:16px 20px;border-radius:17px;border:1px solid #dceae9;margin:12px 0}

@media(max-width:640px){.block-container{padding:0.65rem 0.8rem 2rem}.hero{padding:20px 18px;border-radius:18px}.hero h1{font-size:1.6rem}.insight{padding:13px 14px}}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>✦ Shekhar Health360 AI</h1><p>Predict · Correlate · Personalize &nbsp; | &nbsp; Connected metabolic health intelligence</p></div>',unsafe_allow_html=True)
st.caption('2026 simulated daily/CGM data including Q4. Only Q1–Q3 A1c and Oct 20 weight are user-reported; Q4 is a future illustrative scenario. Not for clinical decisions.')

@st.cache_data
def create_data():
    rng=np.random.default_rng(20261020)
    dates=pd.date_range('2026-01-01','2026-12-31',freq='D'); n=len(dates)
    day=np.arange(n); quarter=dates.quarter.to_numpy()
    sleep=np.clip(6.35+0.26*(quarter-1)+0.52*np.sin(day/11)+rng.normal(0,.62,n),4.6,9.1)
    steps=np.clip(5100+850*(quarter-1)+1450*np.sin(day/8)+rng.normal(0,1750,n),1300,14500).astype(int)
    readiness=np.clip(75+4.8*(sleep-7)+.0008*(steps-6500)+rng.normal(0,7,n),38,98)
    hrv=np.clip(34+3.6*(sleep-7)+rng.normal(0,6,n),15,72)
    rhr=np.clip(65-1.3*(sleep-7)+rng.normal(0,2.5,n),52,78)
    # Illustrative Oura-inspired signals; not actual Oura exports or official metabolic age.
    stress_minutes=np.clip(38+8*(7.5-sleep)+rng.normal(0,8,n),12,59)
    weight=np.clip(154-11.3*day/(n-1)+1.2*np.sin(day/16)+rng.normal(0,.75,n),140,155)
    resilience=np.clip(67+3.2*(sleep-7.5)+0.12*(hrv-34)-0.045*(stress_minutes-38)+rng.normal(0,4,n),30,95)
    metabolic_age=np.clip(44+0.14*(weight-147)+0.025*(stress_minutes-38)-0.10*(readiness-75)+rng.normal(0,0.8,n),35,55)
    systolic=np.clip(123+.21*(weight-147)-.00042*(steps-6500)+rng.normal(0,5,n),104,139)
    diastolic=np.clip(80+.13*(weight-147)-.0002*(steps-6500)+rng.normal(0,3.5,n),67,91)
    # Meals are illustrative logs, NOT inferred from CGM.
    rice=(rng.random(n)<.48).astype(int)
    sweets=(rng.random(n)<.34).astype(int)
    walk=(steps>7200).astype(int)
    daily=pd.DataFrame({'date':dates,'sleep_hours':sleep,'steps':steps,'readiness':readiness,'hrv_ms':hrv,'resting_hr':rhr,'stress_minutes':stress_minutes,'resilience_score':resilience,'metabolic_age_est':metabolic_age,'weight_lb':weight,'systolic':systolic,'diastolic':diastolic,'rice_dinner_log':rice,'sweet_snack_log':sweets,'activity_day':walk})
    # 15-minute simulated Libre-style sensor readings, with deliberately related signals.
    ts=pd.date_range('2026-01-01','2026-12-31 23:45',freq='15min')
    dindex=(ts.normalize()-dates[0]).days.to_numpy(); h=ts.hour.to_numpy()+ts.minute.to_numpy()/60
    baseline=np.select([quarter[dindex]==1,quarter[dindex]==2,quarter[dindex]==3,quarter[dindex]==4],[172,160,149,140])
    breakfast=21*np.exp(-((h-8.5)/1.05)**2)
    lunch=29*np.exp(-((h-13.0)/1.25)**2)
    dinner=(33+13*rice[dindex]-7*walk[dindex])*np.exp(-((h-20)/1.45)**2)
    snack=14*sweets[dindex]*np.exp(-((h-16.5)/.85)**2)
    glucose=np.clip(baseline-16+breakfast+lunch+dinner+snack+3.2*(7-sleep[dindex])+3*np.sin(2*np.pi*h/24)+rng.normal(0,7.5,len(ts)),54,305)
    cgm=pd.DataFrame({'time':ts,'glucose':glucose,'date':ts.normalize()})
    gd=cgm.groupby('date').glucose.agg(['mean','std','max']).reset_index().rename(columns={'mean':'mean_glucose','std':'glucose_std','max':'peak_glucose'})
    gd['tir']=cgm.assign(in_range=cgm.glucose.between(70,180,inclusive='both')).groupby('date').in_range.mean().to_numpy()*100
    daily=daily.merge(gd,on='date')
    daily['next_morning_glucose']=cgm[cgm.time.dt.hour.between(6,10)].groupby('date').glucose.mean().to_numpy()
    return daily,cgm

daily,cgm=create_data()
st.sidebar.header('Explore health trends')
st.sidebar.caption('Each tab has its own period selector. Q4 is a simulated future scenario, not recorded history.')
st.sidebar.markdown('**Data sources**')
st.sidebar.caption('Synthetic Libre-style CGM · BP · Oura-inspired sleep and activity · Weight')

PERIODS=['Q1 2026','Q2 2026','Q3 2026','Q4 2026 (simulated)','Weekly (latest Q4)','Monthly (latest Q4)','Yearly (2026 simulated)']
def select_period(tab_key):
    choice=st.selectbox('📅 Choose the period to explore',PERIODS,index=2,key=f'period_{tab_key}')
    if choice.startswith('Q'):
        q=int(choice[1]); subset=daily[daily.date.dt.quarter==q].copy()
    elif choice.startswith('Weekly'): subset=daily.tail(7).copy()
    elif choice.startswith('Monthly'): subset=daily.tail(30).copy()
    else: subset=daily.copy()
    csubset=cgm[(cgm.date>=subset.date.min())&(cgm.date<=subset.date.max())].copy()
    st.caption(f'Viewing {subset.date.min():%b %d} – {subset.date.max():%b %d, %Y} · {len(subset)} days'+(' · Q4 values simulated' if (subset.date.dt.quarter==4).any() else ''))
    return choice,subset,csubset

def card(kind,headline,detail,action,stat=''):
    st.markdown(f'<div class="insight {kind}"><div class="tag">{stat or "Pattern to explore"}</div><h4>{headline}</h4><p>{detail}</p><p><b>Try this:</b> {action}</p></div>',unsafe_allow_html=True)

def chart_style(fig,height=320):
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=25,b=18),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#425d72'),legend=dict(orientation='h',y=1.14),hovermode='x unified')
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#e8eef5');return fig

tabs=st.tabs(['🏠 Overview','🔗 Correlations','🔮 Predictions','✨ Health Coach','📱 Mobile & Data'])
with tabs[0]:
    period,d,cg=select_period('overview')
    st.markdown(f'<div class="metric-strip"><b>📅 {period} health snapshot</b> &nbsp; · &nbsp; {len(d)} days of illustrative records.</div>',unsafe_allow_html=True)
    # Transparent illustrative wellness index, not a validated medical score.
    # Every component is calculated from the currently selected period.
    tir_value=float(cg.glucose.between(70,180).mean()*100)
    sleep_value=float(d.sleep_hours.mean())
    sys_value=float(d.systolic.mean()); dia_value=float(d.diastolic.mean())
    steps_value=float(d.steps.mean()); readiness_value=float(d.readiness.mean())
    glucose_score=np.clip(tir_value,0,100)
    sleep_score=np.clip(100-22*max(0,7.5-sleep_value),0,100)
    bp_score=np.clip(100-1.5*max(0,sys_value-120)-2*max(0,dia_value-80),0,100)
    activity_score=np.clip(steps_value/8000*100,0,100)
    recovery_score=np.clip(readiness_value,0,100)
    wellness_score=int(round(.30*glucose_score+.25*sleep_score+.15*bp_score+.15*activity_score+.15*recovery_score))
    if wellness_score>=85: label,accent,soft='Great','#13845f','#e8f7ef'
    elif wellness_score>=70: label,accent,soft='Good','#287fbd','#eaf4ff'
    elif wellness_score>=55: label,accent,soft='OK','#ac7419','#fff4df'
    else: label,accent,soft='Needs Attention','#bd4848','#fff0ef'
    st.markdown(f"""<div style="background:{soft};border:1px solid {accent}55;border-radius:20px;padding:20px 22px;margin:12px 0 18px">
      <div style="font-size:.8rem;font-weight:700;color:#52667a;letter-spacing:.07em">OVERALL WELLNESS STATUS</div>
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:8px 0">
        <div style="font-size:2rem;font-weight:800;color:{accent}">{label}</div>
        <div style="font-size:1.5rem;font-weight:800;color:#173149">{wellness_score}<span style="font-size:.9rem;font-weight:500">/100</span></div>
      </div><div style="height:10px;background:#dce6e9;border-radius:20px;overflow:hidden"><div style="width:{wellness_score}%;height:100%;background:{accent};border-radius:20px"></div></div>
      <div style="font-size:.79rem;color:#52667a;margin-top:10px">Calculated for the selected period · educational composite, not a clinical health assessment</div>
    </div>""",unsafe_allow_html=True)
    st.caption('Status changes when the underlying period averages cross the published thresholds. Colors are not randomly assigned.')
    # Daily composite trend: show whether a period genuinely changes, not arbitrary status changes.
    trend=d.copy()
    trend['glucose_component']=trend.tir.clip(0,100)
    trend['sleep_component']=(100-22*(7.5-trend.sleep_hours).clip(lower=0)).clip(0,100)
    trend['bp_component']=(100-1.5*(trend.systolic-120).clip(lower=0)-2*(trend.diastolic-80).clip(lower=0)).clip(0,100)
    trend['activity_component']=(trend.steps/8000*100).clip(0,100)
    trend['composite']=(.30*trend.glucose_component+.25*trend.sleep_component+.15*trend.bp_component+.15*trend.activity_component+.15*trend.readiness.clip(0,100))
    trend['7-day wellness trend']=trend.composite.rolling(7,min_periods=1).mean()
    fig_status=px.line(trend,x='date',y='7-day wellness trend',labels={'date':'Date','7-day wellness trend':'Wellness score / 100'},title='How your wellness score changes over time')
    fig_status.update_traces(line=dict(color='#0e978a',width=3))
    fig_status.add_hline(y=85,line_dash='dot',line_color='#15906e',annotation_text='Great ≥85')
    fig_status.add_hline(y=70,line_dash='dot',line_color='#347fb4',annotation_text='Good ≥70')
    fig_status.add_hline(y=55,line_dash='dot',line_color='#c58b30',annotation_text='OK ≥55')
    st.plotly_chart(chart_style(fig_status,270),use_container_width=True)
    st.subheader('Your health at a glance')
    # Four meaningful, compact scorecards. No invented overall clinical health score.
    tir_pct=cg.glucose.between(70,180).mean()*100
    bp_sys=d.systolic.mean(); bp_dia=d.diastolic.mean()
    sleep_avg=d.sleep_hours.mean(); step_avg=d.steps.mean()
    def scorecard(label,value,detail,color):
        st.markdown(f"""<div class="scorecard"><div class="eyebrow">{label}</div><div class="big">{value}</div><div class="hint">{detail}</div><div class="stripe" style="background:{color}"></div></div>""",unsafe_allow_html=True)
    a,b,c,e=st.columns(4)
    with a: scorecard('🩸 Glucose in range',f'{tir_pct:.0f}%','70–180 mg/dL · selected period','#19a781' if tir_pct>=70 else '#e6a43c')
    with b: scorecard('❤️ Blood pressure',f'{bp_sys:.0f}/{bp_dia:.0f}','mmHg · period average','#328ac6')
    with c: scorecard('🌙 Sleep' if sleep_avg>=7 else '🌙 Sleep · needs attention',f'{sleep_avg:.1f} h','Explore consistency and duration','#e6a43c' if sleep_avg<7 else '#8c77d5')
    with e: scorecard('⚖️ Weight','147 lb','Current · Oct 20, 2026','#17a38f')
    st.markdown('#### Recovery and wellbeing')
    s1,s2,s3,s4=st.columns(4)
    with s1: scorecard('🧠 Daytime stress',f'{d.stress_minutes.mean():.0f} min','Illustrative stress per day','#e28c42')
    with s2: scorecard('🛡️ Resilience',f'{d.resilience_score.mean():.0f}/100','Illustrative index · not Oura score','#8c77d5')
    with s3: scorecard('🧬 Metabolic age',f'{d.metabolic_age_est.mean():.1f} y','Illustrative estimate · not a clinical metric','#328ac6')
    with s4: scorecard('💜 Readiness',f'{d.readiness.mean():.0f}/100','Illustrative Oura-style score','#17a38f')
    st.markdown('<div style="height:12px"></div>',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.metric('Latest reported A1c (Q3)','7.1%','−0.9 points vs Q1; no Q4 result')
    y.metric('Average glucose',f'{cg.glucose.mean():.0f} mg/dL')
    z.metric('Daily steps',f'{step_avg:,.0f}')
    if sleep_avg < 7:
        card('amber','Sleep needs attention',f'Average sleep is {sleep_avg:.1f} hours per night during the selected period.','Test a consistent bedtime for one week; review sleep and next-morning glucose together.','Sleep and recovery')
    st.markdown('### ✨ What deserves attention')
    st.markdown('<div class="section-note">Simple observations from the selected period. Food-specific suggestions require meal logs.</div>',unsafe_allow_html=True)
    eve=cg[cg.time.dt.hour.between(19,22)].glucose.mean()
    morn=cg[cg.time.dt.hour.between(6,10)].glucose.mean()
    if pd.notna(eve) and pd.notna(morn) and eve>morn+10:
        card('amber','Evening glucose is higher',f'Evening readings average {eve:.0f} mg/dL, compared with {morn:.0f} mg/dL in the morning.','Log dinner portions and compare similar meals; a short post-meal walk may be worth discussing with your care team.','Glucose pattern')
    else:
        card('green','Review your daily glucose pattern',f'Evening average: {eve:.0f} mg/dL; morning average: {morn:.0f} mg/dL.','Keep tracking meals and activity to understand day-to-day differences.','Glucose pattern')
    st.subheader('Recovery and stress trends')
    rec=go.Figure()
    rec.add_trace(go.Scatter(x=d.date,y=d.resilience_score,name='Resilience index',line=dict(color='#8c77d5')))
    rec.add_trace(go.Scatter(x=d.date,y=d.readiness,name='Readiness score',line=dict(color='#16a17e')))
    st.plotly_chart(chart_style(rec,250),use_container_width=True)
    st.subheader('Reported A1c trend · Q1–Q3 only')
    a1c=pd.DataFrame({'quarter':['Q1','Q2','Q3'],'a1c':[8.0,7.4,7.1]})
    fig=go.Figure(go.Scatter(x=a1c.quarter,y=a1c.a1c,mode='lines+markers+text',text=['8.0%','7.4%','7.1%'],textposition='top center',line=dict(color='#12a18c',width=4),marker=dict(size=13)))
    fig.update_yaxes(range=[6.6,8.5],title='A1c (%)');st.plotly_chart(chart_style(fig,260),use_container_width=True)
    left,right=st.columns(2)
    with left:
        st.markdown('**Glucose trend and target range**')
        g=cg.set_index('time').glucose.resample('D').mean().reset_index()
        fig=go.Figure(go.Scatter(x=g.time,y=g.glucose,mode='lines',line=dict(color='#14a38e',width=2),name='Daily mean'))
        fig.add_hrect(y0=70,y1=180,fillcolor='#19a77a',opacity=.09,line_width=0)
        fig.update_yaxes(title='mg/dL');st.plotly_chart(chart_style(fig),use_container_width=True)
    with right:
        st.markdown('**Weight and blood pressure**')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=d.date,y=d.weight_lb,name='Weight (lb)',line=dict(color='#7c68bd',width=2)))
        fig.add_trace(go.Scatter(x=d.date,y=d.systolic,name='Systolic',line=dict(color='#e0a03c',width=2)))
        fig.add_trace(go.Scatter(x=d.date,y=d.diastolic,name='Diastolic',line=dict(color='#3c90d3',width=2)))
        st.plotly_chart(chart_style(fig),use_container_width=True)
    st.caption('Weight trajectories are simulated; the separately reported October 20 weight is 147 lb.')
    st.markdown('---')
    st.subheader('🔎 Explainability · How your health status is calculated')
    with st.expander('🔎 Why this status? Full score breakdown',expanded=True):
        components=[('Glucose time in range',glucose_score,.30,'Percent of CGM readings from 70–180 mg/dL'),('Sleep',sleep_score,.25,'100 − 22 × hours below 7.5, floored at 0'),('Blood pressure',bp_score,.15,'100 minus penalties for systolic above 120 or diastolic above 80'),('Activity',activity_score,.15,'Average steps / 8,000 × 100, capped at 100'),('Readiness',recovery_score,.15,'Average simulated readiness score')]
        st.dataframe(pd.DataFrame([{'Signal':n,'Signal score':f'{v:.0f}/100','Weight':f'{w:.0%}','Contribution':f'{v*w:.1f}','How measured':why} for n,v,w,why in components]),hide_index=True,use_container_width=True)
        st.write(f'**Formula:** weighted contributions sum to **{wellness_score}/100**. Thresholds: Great ≥85 · Good 70–84 · OK 55–69 · Needs Attention <55.')
        st.caption('Prototype thresholds and weights are heuristic, not clinically validated. Missing or simulated signals can make this score misleading.')

with tabs[1]:
    period,d,cg=select_period('correlations')
    st.subheader('What affects my glucose?')
    st.caption('Simple relationships across sleep, activity, blood pressure and glucose. Based on illustrative measurements.')
    choices={'Sleep duration':'sleep_hours','Daily steps':'steps','Stress minutes':'stress_minutes','Systolic BP':'systolic','Weight':'weight_lb'}
    factor=st.selectbox('Compare glucose with',list(choices),index=0)
    variable=choices[factor]
    valid=d[[variable,'mean_glucose']].dropna()
    r=valid[variable].corr(valid.mean_glucose,method='spearman') if len(valid)>3 else np.nan
    if pd.isna(r):
        relationship='Not enough data'; explanation='Try a longer period.'
    elif abs(r)<.2:
        relationship='Little consistent relationship'; explanation='These two measurements do not move together consistently in this period.'
    elif r<0:
        relationship='Often move in opposite directions'; explanation=f'Higher {factor.lower()} tends to occur with lower glucose in these records.'
    else:
        relationship='Often move in the same direction'; explanation=f'Higher {factor.lower()} tends to occur with higher glucose in these records.'
    st.metric('What the data suggests',relationship)
    st.write(explanation)
    fig=px.scatter(valid,x=variable,y='mean_glucose',labels={variable:factor,'mean_glucose':'Daily average glucose (mg/dL)'},opacity=.6,color_discrete_sequence=['#159e91'])
    st.plotly_chart(chart_style(fig,300),use_container_width=True)
    st.info('Correlation is not proof of cause. The data is simulated; use actual CGM and Oura readings before making personal conclusions.')
    with st.expander('Advanced: correlation strength'):
        st.write(f'Spearman correlation: {r:+.2f}' if pd.notna(r) else 'Not available')
        st.write('Closer to +1 means values rise together; closer to −1 means they move in opposite directions; near 0 means little monotonic relationship.')
with tabs[2]:
    period,d,cg=select_period('predictions')
    st.subheader('What might my glucose look like tomorrow?')
    st.caption('A machine-learning estimate of next-day average glucose, based on illustrative health history.')
    model_data=daily[daily.date<=d.date.max()].copy()
    features_model=['sleep_hours','steps','readiness','hrv_ms','resting_hr','weight_lb','systolic','diastolic','mean_glucose','tir','glucose_std','peak_glucose','stress_minutes']
    X=model_data[features_model].iloc[:-1]
    y=model_data.mean_glucose.shift(-1).iloc[:-1]
    split=int(len(X)*.8)
    model=HistGradientBoostingRegressor(max_iter=95,max_leaf_nodes=13,l2_regularization=4,random_state=7)
    model.fit(X.iloc[:split],y.iloc[:split])
    pred=model.predict(X.iloc[split:])
    mae=mean_absolute_error(y.iloc[split:],pred)
    p=model.predict(model_data[features_model].iloc[[-1]])[0]
    today=model_data.mean_glucose.iloc[-1]
    rmse=float(np.sqrt(mean_squared_error(y.iloc[split:],pred)))
    baseline=X.mean_glucose.iloc[split:]; baseline_mae=mean_absolute_error(y.iloc[split:],baseline)
    direction='Higher' if p>today+5 else ('Lower' if p<today-5 else 'Similar')
    st.markdown('<div class="metric-strip"><b>Tomorrow at a glance</b><br>Forecast is for average glucose over the next day — not a low-glucose alert or medical recommendation.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric('🔮 Forecast',f'{p:.0f} mg/dL',f'{p-today:+.0f} mg/dL vs latest',delta_color='inverse')
    b.metric('📊 Expected direction',direction,help='Higher or lower means a difference greater than 5 mg/dL; an illustrative UI rule.')
    c.metric('🎯 Typical error (MAE)',f'{mae:.1f} mg/dL',help='Mean absolute error on the held-out final 20% of synthetic days. NOT a prediction interval.')
    m1,m2,m3=st.columns(3)
    m1.metric('RMSE',f'{rmse:.1f} mg/dL',help='Root mean squared error penalizes larger misses more than MAE.')
    m2.metric('Test days',f'{len(pred)}',help='Final 20% of dates were held out, in chronological order.')
    m3.metric('Naive baseline MAE',f'{baseline_mae:.1f} mg/dL',help='Predict tomorrow equals the previous actual day. Compare against this simple baseline.')
    st.markdown('#### CGM pattern intelligence')
    latest_week=d.tail(min(7,len(d)))
    last_cg=cg[cg.date.isin(latest_week.date)]
    low_pct=float((last_cg.glucose<70).mean()*100)
    high_pct=float((last_cg.glucose>180).mean()*100)
    variability=float(last_cg.glucose.std()/last_cg.glucose.mean()*100)
    afternoon=float(last_cg[last_cg.time.dt.hour.between(12,17)].glucose.mean())
    evening=float(last_cg[last_cg.time.dt.hour.between(18,23)].glucose.mean())
    m4,m5,m6,m7=st.columns(4)
    m4.metric('🟢 Time in range (7d)',f'{100-low_pct-high_pct:.0f}%',help='Percentage of synthetic 15-minute readings between 70 and 180 mg/dL')
    m5.metric('🟠 Above range (7d)',f'{high_pct:.1f}%',help='Readings above 180 mg/dL; descriptive, not predicted')
    m6.metric('🔵 Below range (7d)',f'{low_pct:.1f}%',help='Readings below 70 mg/dL; descriptive, not predicted')
    m7.metric('〰️ CGM variability (CV)',f'{variability:.1f}%',help='Standard deviation divided by mean, times 100; descriptive')
    st.markdown('#### Typical glucose by hour · last 7 days')
    hourly=last_cg.assign(hour=last_cg.time.dt.hour).groupby('hour',as_index=False).glucose.mean()
    hourfig=px.line(hourly,x='hour',y='glucose',markers=True,labels={'hour':'Hour of day','glucose':'Average glucose (mg/dL)'})
    hourfig.update_traces(line=dict(color='#118c98',width=3))
    hourfig.add_hrect(y0=70,y1=180,fillcolor='#16a57a',opacity=.07,line_width=0)
    st.plotly_chart(chart_style(hourfig,250),use_container_width=True)
    st.caption('The four CGM metrics and hourly profile describe historical simulated readings. Only next-day mean glucose is predicted by ML; future time in range, spikes and lows are NOT predicted.')
    if mae>=baseline_mae: st.warning('⚠️ The ML model does not beat the naive baseline on this simulated holdout. Treat the forecast as a demonstration only.')
    else: st.success('The ML model beats the previous-day baseline on this simulated holdout; this does not establish clinical usefulness.')
    if p>today+5:
        st.warning('Predicted average is higher than the latest day. Review meals, activity and sleep patterns; this is not a medical alert.')
    elif p<today-5:
        st.success('Predicted average is lower than the latest day. Continue tracking the pattern.')
    else:
        st.info('Predicted average is similar to the latest day. Keep tracking to identify consistent patterns.')
    fig=go.Figure()
    test_dates=model_data.date.iloc[split:-1]
    fig.add_trace(go.Scatter(x=test_dates,y=y.iloc[split:],name='Actual',line=dict(color='#157e9d',width=2)))
    fig.add_trace(go.Scatter(x=test_dates,y=pred,name='Predicted',line=dict(color='#dc963e',width=2,dash='dash')))
    fig.update_yaxes(title='Daily average glucose (mg/dL)')
    st.plotly_chart(chart_style(fig,310),use_container_width=True)
    st.caption('Blue = synthetic reference readings; orange = model forecasts. Model performance is not validated on real patients.')
    with st.expander('🔎 Explainability: where the prediction comes from',expanded=True):
        st.write('**Model:** scikit-learn HistGradientBoostingRegressor. **Target:** next-day mean glucose (mg/dL). **Training:** first 80% of dated records; **test:** final 20%. No random shuffle.')
        st.write('**Inputs:** current-day mean, time in range, peak and variability of CGM; sleep, steps, stress, readiness, HRV, resting HR, weight and blood pressure.')
        st.write('**MAE** is the average absolute miss; **RMSE** emphasizes larger misses. The model is compared with a simple previous-day baseline.')
        st.caption('Training is limited to the selected period end to avoid using future quarters in historical forecasts. These inputs are features, not proven causes. This model does not provide calibrated confidence intervals, feature-level explanations, or clinical safety validation.')
    with st.expander('How does the prediction work?'):
        st.write('A gradient-boosted regression model learns patterns in daily glucose, sleep, activity, recovery, BP and weight. The last 20% of dates are held out for testing. This predicts a daily average, not sudden glucose lows or spikes.')
    st.warning('For presentation only. Do not use forecasts for insulin, medication or treatment decisions.')
with tabs[3]:
    period,d,cg=select_period('coach')
    st.subheader('✨ Your 7-day Health Coach Plan')
    st.caption('Prioritized, explainable lifestyle experiments using simulated patterns. These are not LLM-generated clinical recommendations.')
    ev=float(cg[cg.time.dt.hour.between(19,22)].glucose.mean()); am=float(cg[cg.time.dt.hour.between(6,10)].glucose.mean())
    sleep_avg=float(d.sleep_hours.mean()); step_avg=float(d.steps.mean()); stress_avg=float(d.stress_minutes.mean())
    focus=[]
    if sleep_avg<7.2: focus.append(('1','🌙 Improve sleep consistency','Aim for a consistent bedtime and record 7 nights of sleep.','Sleep average',f'{sleep_avg:.1f} h/night','Sleep below 7.2 h was selected as a demonstration priority.'))
    if ev>am+10: focus.append((str(len(focus)+1),'🍽️ Understand evening glucose','Log dinner timing and portion; compare CGM curves after similar dinners.','Evening vs morning',f'{ev:.0f} vs {am:.0f} mg/dL','Evening CGM average exceeds morning by more than 10 mg/dL; this does not identify a specific food.'))
    if step_avg<7500: focus.append((str(len(focus)+1),'🚶 Add gentle movement','If appropriate for you, try a 10–15 minute walk after dinner and track response.','Daily steps',f'{step_avg:,.0f}','Average steps are below the illustrative 7,500-step threshold.'))
    if stress_avg>40: focus.append((str(len(focus)+1),'🧘 Make time for recovery','Try a short relaxation break; compare sleep and glucose across similar days.','Daily stress',f'{stress_avg:.0f} min','Average simulated stress is above the 40-minute demo threshold.'))
    if not focus: focus.append(('1','🌱 Maintain healthy routines','Continue tracking sleep, activity, and glucose patterns.','Current trends','No priority trigger','No illustrative priority thresholds were exceeded.'))
    st.markdown('#### Your top priorities')
    for number,title,action,metric,value,reason in focus[:3]:
        with st.container(border=True):
            st.markdown(f'**{number}. {title}**')
            st.write(f'**Try:** {action}')
            st.caption(f'📊 Evidence: {metric} = {value}')
            with st.expander('🔎 Why this recommendation?'):
                st.write(reason)
                st.write('How to evaluate: log the experiment, compare similar days, and review whether the measured pattern repeats. Other factors may explain the association.')
    st.markdown('#### 7-day action tracker')
    for item in ['Record sleep and wake time','Log dinner timing and meals','Record steps and movement','Review glucose trends after 7 days']:
        st.checkbox(item,key='coach_'+item)
    st.markdown('#### How the coach chooses priorities')
    with st.expander('🔎 Transparent decision rules',expanded=True):
        st.write('Rules use period averages and thresholds: sleep <7.2 h; evening glucose > morning by 10 mg/dL; steps <7,500/day; stress >40 min/day. Only triggered recommendations appear. These are demonstration thresholds, not clinical standards.')
        st.write('Evidence shown beside each recommendation comes from the selected period. No claim of causation is made. Food-specific insights require meal logs.')
    st.warning('Educational prototype with simulated data. Consult a qualified clinician before changing treatment, medication, or exercise when medically restricted.')
with tabs[4]:
    period,d,cg=select_period('data')
    st.subheader('Mobile access')
    st.write('On your Mac, run `python -m streamlit run app.py --server.address 0.0.0.0`. Open the **Network URL** shown in Terminal on your phone while both devices are on the same Wi-Fi. Keep the Mac awake.')
    st.write('Your previous network address was `http://192.168.1.81:8501`; use the address Streamlit currently displays if it changes.')
    st.subheader('Export and inspect the data')
    st.download_button('Download combined 12-month illustrative dataset',daily.to_csv(index=False).encode(),file_name='health360_simulated_daily_2026.csv',mime='text/csv')
    with st.expander('Preview combined daily data'):
        st.dataframe(daily.tail(20),use_container_width=True)
    st.caption('Stress, resilience and metabolic age here are illustrative Oura-inspired fields, not official Oura measurements or validated estimates. No external health accounts are connected. This prototype does not retrieve live Libre or Oura readings. Do not publish personal health information in a public GitHub repository.')
