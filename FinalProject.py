# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:37:42 2023

@author: Garrett Ramos
"""

import arviz as az
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from FBref_HeadshotScraping import get_player_url, get_player_headshot, save_headshot_jpg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

df = pd.read_csv('MLS_PlayerStats.csv')
nycfc = df[df['Squad'] == 'NYCFC'].fillna(0)
nycfc = nycfc[nycfc['G+A'] > 0]
nycfc = nycfc.iloc[:, :24]
nycfc['xA+G'] = nycfc['xG'] + nycfc['xAG']
nycfc_xAG = nycfc['xA+G']
nycfc_xAG = nycfc_xAG.reset_index(drop=True)
nycfc_x = nycfc.iloc[:, 9:24]
nycfc_y = nycfc.iloc[:, 4]
nycfc_y = nycfc_y.reset_index(drop=True)

data = df[df['Squad'] != 'NYCFC'].fillna(0)
#data = data[data['Min'] > 125]
data = data[data['G+A'] > 0]
y_data = data['G+A'].to_numpy()
x_data = data.iloc[:, 9:24]


with pm.Model() as model:
    Sh90 = pm.MutableData("Sh/90", x_data['Sh/90'])
    SoT90 = pm.MutableData("SoT/90", x_data['SoT/90'])
    KP = pm.MutableData("KP", x_data['KP'])
    FinalThird = pm.MutableData("Final 3rd", x_data['Final 3rd'])
    PPA = pm.MutableData("PPA", x_data['PPA'])
    CrsPA = pm.MutableData("CrsPA", x_data['CrsPA'])
    PrgP = pm.MutableData("PrgP", x_data['PrgP'])
    SCA90 = pm.MutableData("SCA90", x_data['SCA90'])
    AttThird = pm.MutableData("Att 3rd", x_data['Att 3rd'])
    AttPen = pm.MutableData("Att Pen", x_data['Att Pen'])
    Live = pm.MutableData("Live", x_data['Live'])
    Succ = pm.MutableData("Succ", x_data['Succ'])
    FinalThirdCarr = pm.MutableData("Final 3rd Carries", x_data['Final 3rd Carries'])
    CPA = pm.MutableData("CPA", x_data['CPA'])
    PrgR = pm.MutableData("PrgR", x_data['PrgR'])
    
    dat_y = pm.Data("y", y_data, mutable=True)
    
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta0 = pm.Normal('beta0', mu=x_data['Sh/90'].mean(), sigma=x_data['Sh/90'].std())
    beta1 = pm.Normal('beta1', mu=x_data['SoT/90'].mean(), sigma=x_data['SoT/90'].std())
    beta2 = pm.Normal('beta2', mu=x_data['KP'].mean(), sigma=x_data['KP'].std())
    beta3 = pm.Normal('beta3', mu=x_data['Final 3rd'].mean(), sigma=x_data['Final 3rd'].std())
    beta4 = pm.Normal('beta4', mu=x_data['PPA'].mean(), sigma=x_data['PPA'].std())
    beta5 = pm.Normal('beta5', mu=x_data['CrsPA'].mean(), sigma=x_data['CrsPA'].std())
    beta6 = pm.Normal('beta6', mu=x_data['PrgP'].mean(), sigma=x_data['PrgP'].std())
    beta7 = pm.Normal('beta7', mu=x_data['SCA90'].mean(), sigma=x_data['SCA90'].std())
    beta8 = pm.Normal('beta8', mu=x_data['Att 3rd'].mean(), sigma=x_data['Att 3rd'].std())
    beta9 = pm.Normal('beta9', mu=x_data['Att Pen'].mean(), sigma=x_data['Att Pen'].std())
    beta10 = pm.Normal('beta10', mu=x_data['Live'].mean(), sigma=x_data['Live'].std())
    beta11 = pm.Normal('beta11', mu=x_data['Succ'].mean(), sigma=x_data['Succ'].std())
    beta12 = pm.Normal('beta12', mu=x_data['Final 3rd Carries'].mean(), sigma=x_data['Final 3rd Carries'].std())
    beta13 = pm.Normal('beta13', mu=x_data['CPA'].mean(), sigma=x_data['CPA'].std())
    beta14 = pm.Normal('beta14', mu=x_data['PrgR'].mean(), sigma=x_data['PrgR'].std())
    sigma = pm.HalfCauchy('sigma', beta=1)
    
    mu = alpha + beta0*Sh90 + beta1*SoT90 +\
        beta2*KP + beta3*FinalThird + beta4*PPA + beta5*CrsPA +\
            beta6*PrgP + beta7*SCA90 + beta8*AttThird + beta9*AttPen +\
                beta10*Live + beta11*Succ + beta12*FinalThirdCarr +\
                    beta13*CPA + beta14*PrgR

    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=dat_y)

    trace = pm.sample(10000)

az.plot_trace(trace)
summary = az.summary(trace)
post_plot = az.plot_posterior(trace)

ppc = pm.sample_posterior_predictive(trace, model=model, predictions=True)
pred_summary = az.summary(ppc.predictions, hdi_prob=0.95)

RSS = ((y_data - pred_summary['mean'].values)**2).sum()
TSS = ((y_data - y_data.mean())**2).sum()
R2 = 1 - (RSS / TSS)

slope, intercept = np.polyfit(y_data, pred_summary['mean'].values, 1)
line_fit = slope * y_data + intercept
plt.scatter(y_data, pred_summary['mean'].values, alpha=0.5)
plt.plot(y_data, line_fit, color='red')
plt.title('Goodness of Fit')
plt.xlabel('Actual G+A/90')
plt.ylabel('Predicted G+A/90')
plt.text(0.75, 0.95, f'R-squared = {R2:.2f}', transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='grey', alpha=0.7))
plt.xlim(0, 3)
plt.ylim(0, 3)
#plt.show()
plt.savefig('R2_plot.png')

with model:
    pm.set_data({"Sh/90": np.array(nycfc_x['Sh/90']),\
                 "SoT/90": np.array(nycfc_x['SoT/90']),"KP": np.array(nycfc_x['KP']),\
                    "Final 3rd": np.array(nycfc_x['Final 3rd']),"PPA": np.array(nycfc_x['PPA']),\
                    "CrsPA": np.array(nycfc_x['CrsPA']),"PrgP": np.array(nycfc_x['PrgP']),\
                    "SCA90": np.array(nycfc_x['SCA90']),"Att 3rd": np.array(nycfc_x['Att 3rd']),\
                    "Att Pen": np.array(nycfc_x['Att Pen']),"Live": np.array(nycfc_x['Live']),\
                        "Succ": np.array(nycfc_x['Succ']),"Final 3rd Carries": np.array(nycfc_x['Final 3rd Carries']),\
                            "CPA": np.array(nycfc_x['CPA']),"PrgR": np.array(nycfc_x['PrgR']), "y": nycfc_y})
    trace.extend(pm.sample_posterior_predictive(trace))
    
nycfc_pred = pd.Series(trace.posterior_predictive["obs"].mean(dim=["chain", "draw"]).values, name='pred')
nycfc_names = nycfc['Player']
nycfc_names = nycfc_names.reset_index(drop=True)
nycfc_final = pd.DataFrame({'Name': nycfc_names, 'GA': nycfc_y, 'Pred GA': nycfc_pred, 'xA+G': nycfc_xAG})
df_ = nycfc_final.iloc[[0,1,3,4,5]]
nycfc_final['Photo'] = ''

### Scraping player headshots
# for index, row in df_.iterrows():
#     url = get_player_url(row['Name'])
#     print(url)
#     hs = get_player_headshot(url)
#     save_headshot_jpg(hs, url)
    
for index, row in nycfc_final.iterrows():
    nycfc_final.loc[index, 'Photo'] = f'{row["Name"].replace(" ", "-")}.png'
    

fig, ax = plt.subplots(facecolor='lightblue')
ax.scatter(nycfc_final['GA'], nycfc_final['Pred GA'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', c='gray')
ax.set_xlabel('G+A/90')
ax.set_ylabel('Predicted G+A/90')
ax.set_title('2023 NYCFC: Actual G+A vs. Predicted G+A', fontsize=11)
ax.text(s='Outfield players with at least 1 G+A', x=0.51, y=0.205, fontsize=7, fontstyle='italic')
ax.text(s='Data from FBRef', x=0.05, y=0.05, fontsize=7, fontstyle='italic')
ax.grid(True)

for i in range(len(nycfc_final)):
    img = OffsetImage(plt.imread(nycfc_final['Photo'][i]), zoom=0.225)
    ab = AnnotationBbox(img, xy=(nycfc_final['GA'][i], nycfc_final['Pred GA'][i]), frameon=False)
    ax.add_artist(ab)
#plt.show()
plt.savefig('mainViz.png')




