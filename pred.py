import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
from sklearn import svm
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase
import gc

def eval(year, n_feature_elimination=10, SPCDs=[131,110,111,121]):
	path = 'plot/base/'+year
	if not os.path.exists(path):
		os.makedirs(path)

	path = 'result/base/'+year
	if not os.path.exists(path):
		os.makedirs(path)
	
	y_data = pd.read_csv('dataset/presence_data.csv')
	x_data = pd.read_csv('dataset/feature_' + year + '.csv')
	x_data.drop(x_data.columns[0], axis=1,inplace=True)
	join_data = pd.concat([y_data, x_data], axis=1)

	# Outlier Elimination
	feature_names = join_data.columns[6:].tolist()
	join_data = join_data[(np.abs(stats.zscore(join_data[feature_names])) < 3).all(axis=1)]

	x_data = join_data.iloc[:,6:]
	y_data = join_data.iloc[:,:6]

	# Normalization
	mean = x_data.mean(axis=0)
	std = x_data.std(axis=0)
	x_data = (x_data - mean) / std

	vif_data = pd.DataFrame() 
	vif_data["feature"] = x_data.columns
	vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) 
		           for i in range(len(x_data.columns))] 
	if year == '1991_2020':
		n_feature_elimination -= 1
	f = open('result/base/VIF_'+ year +'.txt','w')
	print(vif_data, file = f)
	for _ in range(n_feature_elimination):
	    largest_vif_id = vif_data['VIF'].idxmax()
	    x_data.drop(x_data.columns[largest_vif_id], axis=1,inplace=True)
	    vif_data = pd.DataFrame() 
	    vif_data["feature"] = x_data.columns
	    vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) 
		               for i in range(len(x_data.columns))] 
	print('----------------------------------------\n', file = f)
	print(vif_data, file = f)
	f.close()
	join_data = pd.concat([y_data, x_data], axis=1)
	species_name = {131: "loblolly pine", 111: "slash pine", 110: "shortleaf pine", 121: "longleaf pine"}
	species_n = {111:14000, 110:24000, 121:10000}
	for SPCD in SPCDs:
		with open("dataset/background_"+str(SPCD), "r") as fp:
		    non_species = json.load(fp)
		    
		
		species_data = join_data[join_data['SPCD'] == SPCD]
		non_species_data = join_data[join_data['index'].isin(non_species)]
		non_species_data_a = non_species_data.assign(PRESENCE = 0)

		if SPCD == 131:
		    species_data_a = species_data.sample(n=20000, random_state = 42)
		    join_data_a = pd.concat([species_data_a, non_species_data_a], axis=0)
		else:
		    non_species_data_b = non_species_data_a.sample(n=species_n[SPCD], random_state = 42)
		    join_data_a = pd.concat([species_data, non_species_data_b], axis=0)


		x_data = join_data_a.iloc[:,6:]
		y_data = join_data_a.iloc[:,5]
		
		
		
		X_train, X_test, y_train, y_test = train_test_split(
		    x_data, y_data, test_size=0.3, random_state=42, stratify=y_data)

		rf = RandomForestClassifier()
		rf.fit(X_train, y_train)
		
		
		feature_names = X_train.columns
		mdi_importances = pd.Series(
		    rf.feature_importances_, index=feature_names
		).sort_values(ascending=True)
		mdi_importances.to_csv('result/base/'+year+'/'+year+'_'+str(SPCD)+'_MDI.csv')
		ax = mdi_importances.plot.barh()
		ax.set_title("Feature Importances (MDI) for "+ species_name[SPCD] + " with "+year+" data")
		ax.figure.tight_layout()
		ax.figure.savefig('plot/base/'+year+'/'+year+'_'+str(SPCD)+'_MDI.png')
		plt.close()
		topFeatures = mdi_importances.index.tolist()[-6:]


		def f(x, a, b, c, d):
		    return a / (1. + np.exp(-c * (x - d))) + b

		y = rf.predict_proba(X_test)[:,1]

		fig, axs = plt.subplots(3, 2)

		for i in range(6):
			x = X_test[topFeatures[i]].to_numpy()
			try:
				popt, pcov = opt.curve_fit(f, x, y, method="trf")
		    # print(popt)
				x_fit = np.sort(x)
				y_fit = f(x_fit, *popt)

		    
				axs[i//2,i%2].plot(x_fit * std[topFeatures[i]] + mean[topFeatures[i]], y_fit, '-')
			except:
				axs[i//2,i%2].plot(x * std[topFeatures[i]] + mean[topFeatures[i]], y, 'o')
		    
			axs[i//2,i%2].set_xlabel(topFeatures[i])
		    

		# for ax in axs.flat:
		#     ax.set(ylabel=)

		# Hide x labels and tick labels for top plots and y ticks for right plots.
		# for ax in axs.flat:
		#     ax.label_outer()
		fig.supylabel('logistic output(suitability)')
		fig.suptitle("Response Curve of top 6 features for "+ species_name[SPCD] + " with "+year+" data")
		fig.tight_layout()
		plt.savefig('plot/base/'+year+'/'+year+'_'+str(SPCD)+'_response_curve.png')
		# plt.show()
		plt.close()
		
		test_ids = ["126", "245", "370"]
		for test_id in test_ids:
			test_data = pd.read_csv('dataset/predictions_'+test_id +'.csv')
			#test_data.info()
			
			grid_size = 0.1
			test_data['Longitude'] = test_data['Longitude'].round(2)
			test_data['Latitude'] = test_data['Latitude'].round(2)
			longmax = test_data['Longitude'].max()
			longmin = test_data['Longitude'].min()
			latmax = test_data['Latitude'].max()
			latmin = test_data['Latitude'].min()
			# print(longmax, longmin, latmax, latmin)
			longgrid = np.arange(longmin, longmax+grid_size, grid_size)
			latgrid = np.arange(latmin, latmax+grid_size, grid_size)
			X, Y =  np.meshgrid(longgrid, latgrid)
			Z = np.zeros((latgrid.size, longgrid.size))
			Z.fill(-9999)
			
			x_predict = (test_data[x_data.columns]-mean[x_data.columns])/std[x_data.columns]
			# x_predict.info()
			y_predict = rf.predict_proba(x_predict)[:,1]
			record = pd.DataFrame(test_data[['Latitude', 'Longitude']])
			record['Predict'] = y_predict
			record.to_csv('result/base/'+year+'/'+year+'_'+str(SPCD) +'_'+test_id+'_pred.csv',index=False)
			# print(y_predict)
			# x_location = test_data[["Longitude","Latitude"]]
			# x_location.info()
			dis_count_label = ["low-suitable", "medium-suitable","high-suitable"]
			dis_count = [0,0,0]
			row_count = int(y_predict.shape[0]/3)
			y_predict_avg = (y_predict[:row_count] + y_predict[row_count:2*row_count] + y_predict[2*row_count:])/3
			for idx, j in enumerate(y_predict_avg):
			    if j >= 0.2 and j < 0.4:
			    	dis_count[0] += 100
			    if j >= 0.4 and j < 0.6:
			    	dis_count[1] += 100
			    if j >= 0.6:
			    	dis_count[2] += 100
			    ilo = np.searchsorted(longgrid, test_data["Longitude"][idx])
			    ila = np.searchsorted(latgrid, test_data["Latitude"][idx])
			    Z[ila,ilo] = j

			fig = plt.figure()

			m = Basemap(llcrnrlon=longmin,llcrnrlat=latmin,urcrnrlon=longmax,urcrnrlat=latmax,
				    projection='cyl', resolution = 'c')

			m.drawcoastlines(color='blue',linewidth=1)
			m.drawcountries(color='gray',linewidth=1)

			#Fill the continents with the land color
			# m.fillcontinents(color='coral',lake_color='aqua')

			levels = [0.0, 0.2, 0.4, 0.6, 1.0]
			cs = plt.contourf(X, Y, Z, levels=levels, colors=('white','yellow','g','b','r'), extend = 'min')
			# cs.cmap.set_under('w')
			plt.colorbar(cs)
			plt.title('Distribution Map for '+ species_name[SPCD] + ' with '+year+" data" + ' and SSP'+test_id, fontsize = 7)
			plt.savefig('plot/base/'+year+'/'+year+'_'+str(SPCD)+'_'+test_id+'_map.png')
			plt.close()
			
			fig = plt.figure(figsize = (10, 5))
			plt.bar(dis_count_label, dis_count, color =('g','b','r'), 
				width = 0.4)
			area_record = pd.DataFrame(list(zip(dis_count_label, dis_count)))
			area_record.to_csv('result/base/'+year+'/'+year+'_'+str(SPCD)+'_'+test_id+'_area.csv', index=False)
			plt.xlabel("Suitability")
			plt.ylabel("area km$^2$")
			plt.title('Size of Area for '+ species_name[SPCD] + ' with '+year+" data" + ' and SSP'+test_id)
			plt.savefig('plot/base/'+year+'/'+year+'_'+str(SPCD)+'_'+test_id+'_area.png')
			plt.close()
		del x_data, y_data, X_train, X_test, y_train, y_test
		gc.collect()


