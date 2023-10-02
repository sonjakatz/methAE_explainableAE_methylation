import numpy as np
import os
import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('default')


def RFregression(X_train, 
                 y_train, 
                 X_test, 
                 y_test, 
                 name,
                 saveModel=False, 
                 plot=True, 
                 color=np.random.random(3), 
                 title=None):
    ## Remove samples with NA in target
    # Train
    mask_train = ~np.isnan(y_train)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    # Test
    mask_test = ~np.isnan(y_test)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # Train RFregression model
    regr = RandomForestRegressor(random_state=None)
    regr.fit(X_train, y_train)
    featureImportances = regr.feature_importances_

    # Score on Test set
    y_pred = regr.predict(X_test)
    r2 = regr.score(X_test, y_test)  # coefficient of determination
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    if saveModel:
        # save model
        outPath = f"logs/{name}/regression/"
        os.makedirs(outPath, exist_ok=True)
        pickle.dump(regr, open(f"{outPath}/RFregressor.sav", 'wb'))

    if plot:
        plt.rcParams.update({'font.size': 18})
        fig, (ax) = plt.subplots(1, 1,figsize=(8,8))
        ax.scatter(y_test, y_pred, color=color)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="red", linestyle="dashed", alpha=0.5)
        ax.text(55, 10, f"R² = {round(r2.mean(),2)}")
        ax.text(55, 20, f"RMSE = {round(rmse,2)}")        
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        if title: 
            ax.set_title(f"{title}")
        else: 
            ax.set_title(f"{X_train.shape[1]} features")
        ax.set_xlabel("True Age")
        ax.set_ylabel("Predicted Age")

#         ax2.bar(x=range(len(featureImportances)), height=featureImportances, color=color)
#         ax2.set_title("Feature importance")
#         ax2.set_xlabel("# Feature")
        plt.show()

    return regr, featureImportances, r2, rmse