# %%
# pipenv install pandas scipy plotly scikit-learn optuna shap ipywidgets nbformat numpy==2.0

# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_curve, roc_auc_score, f1_score, precision_score, recall_score

import optuna
import shap

# %%
pd.__version__ 

# %%
df = pd.read_csv('dataset.csv')

# %%
df.info()

# %%
df.head()

# %%
df['data_contratacao'] = pd.to_datetime(df['data_contratacao'], format="%Y-%m-%d")
df['data_demissao'] = pd.to_datetime(df['data_demissao'], format='%Y-%m-%d')
df['data_ultimo_feedback'] = pd.to_datetime(df['data_ultimo_feedback'], format='%Y-%m-%d')
df['data_ultimo_aumento'] = pd.to_datetime(df['data_ultimo_aumento'], format='%Y-%m-%d')
df['data_ultima_mudanca_cargo'] = pd.to_datetime(df['data_ultima_mudanca_cargo'], format='%Y-%m-%d')

# %%
df.info()

# %%
df.head()

# %%
df[df['data_contratacao'] == '2020-01-02']

# %%
df.describe()

# %%
df['tempo_empresa'] = df.apply(lambda x: (x['data_demissao'] - x['data_contratacao']).days if x['churn'] == 1 else (pd.Timestamp.now() - x['data_contratacao']).days, axis=1)

# %%
df.head()

# %%
df[df['churn'] == 1].head()

# %%
df['tempo_desde_ultimo_feedback'] = df.apply(lambda x: (pd.Timestamp.now() - x['data_ultimo_feedback']).days, axis=1)

# %%
df.head()

# %%
df.tail()

# %%
df['dias_desde_ultimo_aumento'] = df.apply(lambda x: (pd.Timestamp.now() - x['data_ultimo_aumento']).days, axis=1)

# %%
df['dias_desde_ultima_mudanca_cargo'] = df.apply(lambda x: (pd.Timestamp.now() - x['data_ultima_mudanca_cargo']).days, axis=1)

# %%
df.tail()

# %%
df.drop(columns=['id'], inplace=True, axis=1)

# %%
df.head()

# %% [markdown]
# ### EDA

# %%
df.isnull().sum()

# %%
df['churn'].value_counts()

# %%
fig = px.bar(df['churn'].value_counts() / len(df) * 100, title='Fator de Churn')
fig.show()

# %%
for col in df.select_dtypes(include=['object']).columns:
  print(f"Valores únicos na coluna {col}:", df[col].unique())

# %%
df.select_dtypes(include=['int64', 'float64']).describe()

# %%
for col in df.select_dtypes(include=['int64', 'float64']).columns:
  if col != 'churn':
    fig = px.box(df, x='churn', y=col, title=f'Boxplot da coluna {col}', color='churn')
    fig.show()

# %%
colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[colunas_numericas].corr()

fig = px.imshow(corr_matrix, title='Matriz de Correlação', color_continuous_scale='Viridis', zmin=-1, zmax=1)

fig.update_traces(text=corr_matrix, texttemplate='%{text:.1%}', textfont={'size': 12})
fig.update_layout(width=1000, height=800, title_font=dict(size=14), font=dict(size=10))

fig.show()

# %%
fig = px.scatter_matrix(df, dimensions=colunas_numericas, color='churn', title='Scatter Matrix')
fig.update_layout(width=1200, height=1000, title_font=dict(size=14), font=dict(size=10))

fig.show()

# %%
colunas_categoricas = df.select_dtypes(include=['object']).columns
for col in colunas_categoricas:
  contigency_table = pd.crosstab(df[col], df['churn'])
  chi2, p_value, dof, expected = chi2_contingency(contigency_table)
  print(f'\nTeste Chi-quadrado para {col} vs Churn')
  print(f'p-value: {p_value}')
  if p_value <= 0.05:
    print('Variáveis são dependentes')
  else:
    print('Variáveis são independentes')

# %%
for col in colunas_categoricas:
  fig = px.histogram(df, x=col, color='churn', title=f'Histograma da coluna {col}', barmode='group')
  fig.show()

# %% [markdown]
# ### PREPARACAO DOS DADOS

# %%
colunas_datas = df.select_dtypes(include=['datetime64']).columns
df.drop(columns=colunas_datas, inplace=True)

# %%
df.info()

# %%
X = df.drop(columns=['churn', 'tipo_demissao'])
y = df['churn']

# %%
features_numericas =  X.select_dtypes(include=['int64', 'float64']).columns
features_categoricas = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
  transformers=[
    ('num', StandardScaler(), features_numericas),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
  ]
)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=42) 

# %%
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# %%
print(X_train.shape, X_test.shape)  

# %% [markdown]
# ### TREINAMENTO DO MODELO BASELINE

# %%
rf_model = RandomForestClassifier(
  n_estimators=100, 
  max_depth=20, 
  min_samples_split=2, 
  min_samples_leaf=1, 
  random_state=42,
  max_features='sqrt',
  class_weight='balanced'
)

# %%
rf_model.fit(X_train, y_train)

# %% [markdown]
# ### ANÁLISE DOS RESULTADOS BASELINE

# %%
y_pred = rf_model.predict(X_test)

# %%
len(y_pred)

# %%
y_pred_proba = rf_model.predict_proba(X_test)

# %%
y_pred_proba

# %%
print(classification_report(y_test, y_pred))  

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# %%
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# %%
fig = px.area(
  x=fpr,
  y=tpr,
  title=f'Curva ROC (AUC={roc_auc:.4f})',
  labels=dict(x='Taxa de Falso Positivo', y='Taxa de Verdadeiro Positivo'),
  width=800,
  height=800
)
fig.add_shape(
  type='line', line=dict(dash='dash'),
  x0=0, x1=1, y0=0, y1=1 
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()

# %%
conf_matrix = confusion_matrix(y_test, y_pred)
fig = ConfusionMatrixDisplay(conf_matrix, display_labels=['Não Churn', 'Churn'])
fig.plot()

# %%
print(f'Log Loss: {log_loss(y_test, y_pred_proba)}')

# %% [markdown]
# ### TREINAR MODELO CROSS VALIDATION E TUNING DE HIPERPARAMETROS

# %%
# Dicionario de hiperparâmetros
params_grids = {
  'n_estimators': [100, 200, 300, 400, 500],
  'max_depth': [None, 10, 20, 30, 40, 50],
  'min_samples_split': [2, 5, 10, 20],
  'min_samples_leaf': [1, 2, 4, 5, 10],
}

# %%
rf_model_cv = RandomForestClassifier(random_state=42, class_weight='balanced', max_features='sqrt')
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_model_cv, param_grid=params_grids, cv=k_folds, scoring='recall', verbose=2)

# %%
grid_search.fit(X_train, y_train)

# %%
best_model = grid_search.best_estimator_

# %%
best_params = grid_search.best_params_

# %%
print(best_params)

# %%
best_score = grid_search.best_score_
best_score

# %%
y_pred = best_model.predict(X_test)

# %%
y_ped_proba = best_model.predict_proba(X_test)

# %%
print(classification_report(y_test, y_pred))

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# %%
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# %%
fig = px.area(
  x=fpr,
  y=tpr,
  title=f'Curva ROC (AUC={roc_auc:.4f})',
  labels=dict(x='Taxa de Falso Positivo', y='Taxa de Verdadeiro Positivo'),
  width=800,
  height=800
)
fig.add_shape(
  type='line', line=dict(dash='dash'),
  x0=0, x1=1, y0=0, y1=1 
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()

# %%
conf_matrix = confusion_matrix(y_test, y_pred)
fig = ConfusionMatrixDisplay(conf_matrix, display_labels=['Não Churn', 'Churn'])
fig.plot()

# %% [markdown]
# ### AJUSTAR THRESHOLD

# %%
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

recalls = []

for threshold in thresholds:
  y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
  recall = recall_score(y_test, y_pred_threshold)
  
  recalls.append(recall)

recalls

# %%
df_trheadholds = pd.DataFrame({'thresholds': thresholds, 'recall': recalls})

# %%
df_trheadholds

# %%
y_pred = (y_pred_proba[:, 1] >= 0.1).astype(int)

# %%
print(classification_report(y_test, y_pred))

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# %%
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# %%
fig = px.area(
  x=fpr,
  y=tpr,
  title=f'Curva ROC (AUC={roc_auc:.4f})',
  labels=dict(x='Taxa de Falso Positivo', y='Taxa de Verdadeiro Positivo'),
  width=800,
  height=800
)
fig.add_shape(
  type='line', line=dict(dash='dash'),
  x0=0, x1=1, y0=0, y1=1 
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()

# %%
conf_matrix = confusion_matrix(y_test, y_pred)
fig = ConfusionMatrixDisplay(conf_matrix, display_labels=['Não Churn', 'Churn'])
fig.plot()

# %% [markdown]
# ### OBTER IMPORTANCIA DAS VARIAVEIS

# %%
importancias = best_model.feature_importances_
nomes_features = preprocessor.get_feature_names_out()

df_importancias = pd.DataFrame({'feature': nomes_features, 'importancia': importancias})

# %%
df_importancias = df_importancias.sort_values(by='importancia', ascending=False).reset_index(drop=True)

# %%
df_importancias

# %%
fig = px.bar(df_importancias.head(10), 
             x='importancia', 
             y='feature', 
             title='Importância das Features', 
             orientation='h', 
             color='importancia', 
             color_continuous_scale='Viridis')
fig.show()

# %% [markdown]
# ### VISUALIZAR ARVORE

# %%
def visualiza_arvore(modelo, indice_arvore, max_profundidade=5):
  plt.figure(figsize=(20, 20))
  plot_tree(modelo.estimators_[indice_arvore], 
            max_depth=max_profundidade, 
            filled=True, 
            feature_names=nomes_features,
            class_names=['Não Churn', 'Churn'],
            fontsize=9,
            proportion=True,
            precision=2,
            rounded=True)
  plt.title(f'Árvore {indice_arvore} da Random Forest', fontsize=14)
  plt.tight_layout()
  plt.show()

# %%
for i in range(5):
  visualiza_arvore(best_model, i)

# %% [markdown]
# ### SHAPLEY VALUES

# %%
explainer_class = shap.Explainer(best_model.predict, X_train, feature_names=nomes_features)

# %%
shape_values = explainer_class(X_test)

# %%
shap.plots.bar(shape_values)

# %%
shap.plots.beeswarm(shape_values, max_display=10)

# %% [markdown]
# ### SHAPLEY VALUE DE UMA LINHA ESPECIFICA

# %%
shap.plots.waterfall(shape_values[1])

# %%



