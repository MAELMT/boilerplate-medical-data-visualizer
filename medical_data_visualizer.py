import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Importation des données
df = pd.read_csv("medical_examination.csv")

# 2 - Ajouter la colonne overweight (IMC)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3 - Normalisation de cholesterol et gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4 - Fonction cat plot
def draw_cat_plot():
    # 5 - Transformer les données en format long
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6 - Grouper les données
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 - Créer le graphique
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar").fig

    # 9 - Sauvegarder
    fig.savefig('catplot.png')
    return fig


# 10 - Fonction heat map
def draw_heat_map():
    # 11 - Nettoyer les données
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12 - Matrice de corrélation
    corr = df_heat.corr()

    # 13 - Masque pour la heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Créer la figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 - Tracer la heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    # 16 - Sauvegarder
    fig.savefig('heatmap.png')
    return fig
