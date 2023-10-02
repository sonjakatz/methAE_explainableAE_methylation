import pandas as pd
import numpy as np
import os
import pickle 
import json
import torch
from data.prepareData import prepareDataLoader_fromPkl
from scripts.train_RFregressor import RFregression
import time

def main():
    t0 = time.time()
    '''
    Merge highly connected CpGs of all chromosomes
    '''
    all_cpgs_high = []
    all_cpgs_low = []
    all_cpgs_none = []

    for CHR in [f"chr{i}" for i in range(1,23)]:
        print(CHR)
        PATH_data = "/data/scratch/skatz/PROJECTS/methylnet/1_healthyVAE/data/GSE87571/train_val_test_sets/"
        PATH_model= f"logs/finalModels/{CHR}"
        PATH_perturbation = f"results/{CHR}/perturbations"

        # Load beta matrix
        #with open(os.path.join(PATH_data, "chr22_wholeDataset.pkl"), "rb") as f: whole_dataset = pickle.load(f) #

        # Load CpG connections
        #conn = pd.read_csv(f"{PATH_perturbation}/quantileCutoff_CpGfocus.csv", index_col=0)
        with open(f"{PATH_perturbation}/global_connectivity_groups.pkl", "rb") as f: dic_globalConn = pickle.load(f)
        chr_idx_cpgs_high = dic_globalConn["global_high"]
        chr_idx_cpgs_low = dic_globalConn["global_low"]
        chr_idx_cpgs_none = dic_globalConn["global_none"]
        print(f"High CpGs: {len(chr_idx_cpgs_high)}\nLow CpGs: {len(chr_idx_cpgs_low)}\nNone CpGs: {len(chr_idx_cpgs_none)}\n")

        all_cpgs_high = all_cpgs_high + chr_idx_cpgs_high
        all_cpgs_low = all_cpgs_low + chr_idx_cpgs_low
        all_cpgs_none = all_cpgs_none + chr_idx_cpgs_none

    print(f"\n\nAll Chromosomes - High CpGs: {len(all_cpgs_high)}\nAll Chromosomes - Low CpGs: {len(all_cpgs_low)}\nAll Chromosomes - None CpGs: {len(all_cpgs_none)}\n")



    ''' 
    Bootstrap - Better performance than random?
    '''

    ## Predictions using same number of random CpGs (median of distribution)
    print(f"Number of CpGs connected: {len(all_cpgs_high)}")
    print(f"Number of CpGs connected: {len(all_cpgs_low)}")
    print(f"Number of CpGs connected: {len(all_cpgs_none)}")

    with open(os.path.join(PATH_data, f"train_methyl_array.pkl"), "rb") as f: train_dataset = pickle.load(f) #
    #train_dataset_beta = train_dataset["beta"].loc[:,selCpgs]    
    with open(os.path.join(PATH_data, f"test_methyl_array.pkl"), "rb") as f: test_dataset = pickle.load(f) #
    #test_dataset_beta = test_dataset["beta"].loc[:,selCpgs]

    random_high = []
    random_low = []
    random_none = []
    for i_iter in range(5):

        ############### Globally high CpGs ###############
        i_high = len(all_cpgs_high)
        cpgs_high = np.random.choice(train_dataset["beta"].columns, i_high)

        ## parse only CpGs connected to latent feature 
        X_train_high_random = train_dataset["beta"].loc[:,cpgs_high]
        y_train_high_random = train_dataset["pheno"]["Age"].values
        X_test_high_random = test_dataset["beta"].loc[:,cpgs_high]
        y_test_high_random = test_dataset["pheno"]["Age"].values

        ## predict with those CpGs
        _, _, r2_high_i = RFregression(X_train=X_train_high_random, 
                                y_train=y_train_high_random,
                                X_test=X_test_high_random,
                                y_test=y_test_high_random,
                                name="finalModels/globalCpGs_high",
                                saveModel=False,
                                plot=False)
        random_high.append(r2_high_i)

        ############### Globally low CpGs ###############
        i_low = len(all_cpgs_low)
        cpgs_low = np.random.choice(train_dataset["beta"].columns, i_low)

        ## parse only CpGs connected to latent feature 
        X_train_low_random = train_dataset["beta"].loc[:,cpgs_low]
        y_train_low_random = train_dataset["pheno"]["Age"].values
        X_test_low_random = test_dataset["beta"].loc[:,cpgs_low]
        y_test_low_random = test_dataset["pheno"]["Age"].values

        ## predict with those CpGs
        _, _, r2_low_i = RFregression(X_train=X_train_low_random, 
                                y_train=y_train_low_random,
                                X_test=X_test_low_random,
                                y_test=y_test_low_random,
                                name="finalModels/globalCpGs_low",
                                saveModel=False,
                                plot=False)
        random_low.append(r2_low_i)

        ############### Globally none CpGs ###############
        i_none = len(all_cpgs_none)
        cpgs_none = np.random.choice(train_dataset["beta"].columns, i_none)

        ## parse only CpGs connected to latent feature 
        X_train_none_random = train_dataset["beta"].loc[:,cpgs_none]
        y_train_none_random = train_dataset["pheno"]["Age"].values
        X_test_none_random = test_dataset["beta"].loc[:,cpgs_none]
        y_test_none_random = test_dataset["pheno"]["Age"].values

        ## predict with those CpGs
        _, _, r2_none_i = RFregression(X_train=X_train_none_random, 
                                y_train=y_train_none_random,
                                X_test=X_test_none_random,
                                y_test=y_test_none_random,
                                name="finalModels/globalCpGs_none",
                                saveModel=False,
                                plot=False)
        random_none.append(r2_none_i)

    # Save for later
    dic_bootstrap = dict()
    dic_bootstrap["r2_high"] = random_high
    dic_bootstrap["r2_low"] = random_low
    dic_bootstrap["r2_none"] = random_none
    os.makedirs("results/globalCpGs/prediction/age/", exist_ok=True)
    with open("results/globalCpGs/prediction/age/globalCpGs_regressionBootstrap_r2_6.pkl", "wb") as f: pickle.dump(dic_bootstrap, f)


if __name__ == "__main__":
    main()