# ******************************************
# Ionel Rodriguez 8-764-368
# Modelos Predictivos
# ******************************************

import pandas
import numpy as np
import os
from sklearn import metrics as skm


def load_base_dataframe():
	data_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "VENTAAUTOS.xlsx")
	df = pandas.read_excel(data_path,  engine='openpyxl')

	# Se procede a limpiar y ajustar la base de datos para que esté lista para el análisis
	df = df.replace(np.nan, "")
	
	# Se convierten los nombres de meses a números
	meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
	df["mes_id"] = df.mes.apply(lambda m: meses.index(m.lower().strip())+1)

	# Se crea una nueva columna de fecha utilizando las columnas de año, mes, y el último día de cada mes
	df["fecha"] = pandas.to_datetime({
		"year": df["año"],
		"month": df.mes_id,
		"day": 1,
	}) + pandas.tseries.offsets.MonthEnd(0)

	# Se convierten las marcas y modelos a variables categóricas
	df['modelo'] = df.modelo.astype("category")
	df['modelo_id'] = df.modelo.cat.codes

	df['marca'] = df.marca.astype("category")
	df['marca_id'] = df.marca.cat.codes

	# Por último se filtran los datos solo para los años del estudio 2022 y 2024
	df = df[(df["año"] < 2025) | (df["año"] >= 2022)].sort_values(by=['año', 'mes_id'], ascending=True)
	return df

def remove_outliers(df_source, column):
	ret = pandas.DataFrame()

	for id,m in enumerate(df_source.modelo.cat.categories):
		sub_df = df_source[df_source.modelo_id == id]
		
		data_m, data_std = np.mean(sub_df[column]), np.std(sub_df[column])
		cutoff = data_std * 1.3
		v0, v1 = data_m - cutoff, data_m + cutoff
		sub_df = sub_df[(sub_df[column] > v0) & (sub_df[column] < v1)]
		ret = pandas.concat([ret, sub_df])
	
	return ret

def build_err_details(real, forecast):
	MSE = skm.mean_squared_error(real, forecast) # Mean Squared Error
	RMSE = np.sqrt(MSE) # Root Mean Squared Error
	MAPE = skm.mean_absolute_percentage_error(real, forecast) * 100 # Mean absolute percentage error
	MAD = skm.mean_absolute_error(real, forecast) # Mean absolute error/deviation
	
	return {
		'MAD': MAD,
		'MAPE': MAPE,
		'MSE': MSE,
		'RMSE': RMSE,
		'std_dev': np.std(forecast, axis=0), # Standard Deviation
		'R^2': skm.r2_score(real, forecast), # Coeficient of determination
	}

