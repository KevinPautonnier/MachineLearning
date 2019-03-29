import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

df = pd.read_csv('./data/LeagueofLegends.csv',sep=',')

# get win_team color
df['win_team'] = np.where(df['bResult']==1, 'blue', 'red')

# Create Figure
fig, ax = plt.subplots(1,1, figsize=(16,14))
fig.subplots_adjust(top=0.9)

#transform string array in real array
df['golddiff'] = df['golddiff'].apply(literal_eval)

#get gold diff of the win_team in the middle of the game
golddiff = []
index = 0
for data in df['golddiff']:
    if(df['win_team'][index] == 'red'):
        golddiff.append(-data[int(len(data)/2)])
    else:
        golddiff.append(data[int(len(data)/2)])
    index += 1

#make graph
p2 = plt.subplot2grid((2,4), (0,1), colspan=3)
x = np.sort(golddiff)
y = np.arange(1, len(x) + 1) / len(x)
plt.plot(x,y, marker='.', linestyle='none', color='blue')

#display percent in y axis
yvals = p2.get_yticks()
p2.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

#add title
p2.set_title('Pourcentage de partie gagnée en fonction de la différence d\'argent au milieu de la partie')

plt.tight_layout()
plt.show()