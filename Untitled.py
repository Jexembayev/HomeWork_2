#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
file_path = 'WaitData.Published.xlsx' 
df = pd.read_excel(file_path, sheet_name='F3')

df_clean = df.drop(columns=[col for col in df.columns if col.startswith('x_')])
X = df_clean.drop(columns=['Wait']) 
Y = df_clean['Wait']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

train_residuals = Y_train - Y_pred_train
test_residuals = Y_test - Y_pred_test


mae_train = mean_absolute_error(Y_train, Y_pred_train)
mae_test = mean_absolute_error(Y_test, Y_pred_test)


print(f"Средняя абсолютная ошибка на обучающих данных: {mae_train:.2f}")
print(f"Средняя абсолютная ошибка на тестовых данных: {mae_test:.2f}")

median_abs_residuals_train = train_residuals.abs().median()
print(f"Медиана абсолютных остатков на обучающих данных: {median_abs_residuals_train:.2f}")


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


file_path = 'WaitData.Published.xlsx'
df = pd.read_excel(file_path, sheet_name='F3')

print("Первые строки загруженной таблицы:")
print(df.head())

df_clean = df.drop(columns=[col for col in df.columns if col.startswith('x_')])

print("\nОставшиеся столбцы после удаления 'x_' столбцов:")
print(df_clean.columns)

X = df_clean.drop(columns=['Wait'])
Y = df_clean['Wait'] 

print("\nРазмерность данных X (признаки):", X.shape)
print("Признаки (первые строки):")
print(X.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

mae_train = mean_absolute_error(Y_train, Y_pred_train)
mae_test = mean_absolute_error(Y_test, Y_pred_test)

print(f'\nСредняя абсолютная ошибка на обучающих данных (MAE): {mae_train:.2f}')
print(f'Средняя абсолютная ошибка на тестовых данных (MAE): {mae_test:.2f}')

train_residuals = Y_train - Y_pred_train
median_abs_residuals_train = train_residuals.abs().median()
print(f"Медиана абсолютных остатков на обучающих данных: {median_abs_residuals_train:.2f}")


# In[3]:


if True:
    print('\n> Рекурсивное исключение признаков (RFE):')
    
    from sklearn.feature_selection import RFE
    from itertools import compress
    
    for nFeatures in range(1, 4):
        rfe = RFE(model, n_features_to_select=nFeatures)
        X_rfe = rfe.fit_transform(X, Y) 
        
        model.fit(X_rfe, Y)

        print(f"Число признаков: {nFeatures}")
        print("Выбранные признаки:", rfe.support_)
        print("Ранжирование признаков:", rfe.ranking_)

        cols = list(compress(X.columns, rfe.support_))
        model.fit(X[cols], Y)
        e = abs(Y - model.predict(X[cols])).mean()
        
        print(f"Средняя абсолютная ошибка для {nFeatures}-признаковой модели: {e:.2f}")
        print("Используемые признаки:", cols, '\n')


# In[4]:


best_features = []
remaining_features = list(X.columns)
min_error = float('inf')

for feature in remaining_features:
    model.fit(X[[feature]], Y)
    e = abs(Y - model.predict(X[[feature]])).mean()
    
    if e < min_error:
        min_error = e
        best_feature = feature

best_features.append(best_feature)
remaining_features.remove(best_feature)
print(f"Лучший один признак: {best_feature}, Ошибка: {min_error:.2f}")

for feature in remaining_features:
    model.fit(X[best_features + [feature]], Y)
    e = abs(Y - model.predict(X[best_features + [feature]])).mean()
    
    if e < min_error:
        min_error = e
        best_second_feature = feature

best_features.append(best_second_feature)
remaining_features.remove(best_second_feature)
print(f"Лучшие два признака: {best_features}, Ошибка: {min_error:.2f}")

for feature in remaining_features:
    model.fit(X[best_features + [feature]], Y)
    e = abs(Y - model.predict(X[best_features + [feature]])).mean()
    
    if e < min_error:
        min_error = e
        best_third_feature = feature

best_features.append(best_third_feature)
remaining_features.remove(best_third_feature)
print(f"Лучшие три признака: {best_features}, Ошибка: {min_error:.2f}")


# In[5]:


best_features = []
remaining_features = list(X.columns) 
min_error = float('inf')  
feature_errors = []  

for i in range(15):
    for feature in remaining_features:
        model.fit(X[best_features + [feature]], Y)
        e = abs(Y - model.predict(X[best_features + [feature]])).mean()
        
        if e < min_error:
            min_error = e
            best_feature = feature
    
    best_features.append(best_feature)
    remaining_features.remove(best_feature)

    feature_errors.append((len(best_features), min_error))

    print(f'Количество признаков: {len(best_features)}, Ошибка: {min_error:.4f}')

    if min_error < 24:
        print(f'Ошибка ниже 24 достигнута при {len(best_features)} признаках.')
        break


# In[ ]:





# In[ ]:




