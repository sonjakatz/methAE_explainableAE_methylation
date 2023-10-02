import copy
import pandas as pd
import numpy as np
import pickle 
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", font_scale=1.3)

def latSpacePerturbation_quantileCutoff(model, data_dic, name, save_perturbations=False, plot=False):

    #####################################  PERTURBATIONS  #####################################

    data_tensor = torch.tensor(data_dic["beta"].values, dtype=torch.float32)

    # generate latent space
    with torch.no_grad():
        latSpace_np = model.generate_embedding(data_tensor).detach().numpy()

    # Derive mean and std for perturbations
    mean = latSpace_np.mean(axis=0)
    std = latSpace_np.std(axis=0)

    print("\nPerturbing...\n")

    # Perturb
    dic_pert = {}
    for latvar in range(latSpace_np.shape[1]):
        dic_eachlat = {}
        for traversalVal in [std[latvar], std[latvar]*-1]: # perturb by adding / subtracting 1 std to every value
            with torch.no_grad():
                # generate latent space
                latSpace_orig = model.generate_embedding(data_tensor)
                # perturb
                latSpace_pert = copy.deepcopy(latSpace_orig)
                latSpace_pert[:,latvar] += traversalVal   

                # generate original & perturbed reconstructions
                recon_orig =  model.decode(latSpace_orig)
                recon_pert =  model.decode(latSpace_pert)

                # determine difference between orig & perturbed (median of all patients for each CpG)
                # potential point of improvement: some patients more affected than others?!
                dic_eachlat[str(round(traversalVal,2))] = np.median(abs(recon_orig - recon_pert).detach().numpy(), axis=0)
        dic_pert[latvar] = dic_eachlat

    # Save to file
    if save_perturbations is True:
        outDir = f"results/{name}/perturbations"
        os.makedirs(outDir, exist_ok=True)
        pickle.dump(dic_pert, open(os.path.join(outDir, "perturbations_by_std.pkl"), "wb"))

    #####################################  APPLY THRESHOLDING  #####################################

    df_allLat = pd.DataFrame()
    for i in dic_pert.keys():
        df_allLat[i]=pd.DataFrame.from_dict(dic_pert[i]).median(axis=1)
    # rename index to cpg names
    df_allLat.index = data_dic["beta"].columns

    # save to file
    if save_perturbations is True:
        outDir = f"results/{name}/perturbations"
        os.makedirs(outDir, exist_ok=True)
        df_allLat.to_csv(os.path.join(outDir, "perturbations_by_std.csv"))


    ### derive high, medium, low, and none cpgs for each latent feature
    dic_allLatFeatures = {}
    for latFeature_i in range(len(dic_pert.keys())):
        print(f"Applying threshold for latent feature: {latFeature_i}")
        dic_latFeature = applyStdThreshold(data=df_allLat.iloc[:,latFeature_i],
                                            name=name, 
                                            plot=plot, 
                                            save=True)
        # Save in final dic
        dic_allLatFeatures[latFeature_i] = dic_latFeature


    #####################################  REORDER FOR CPG FOCUS  #####################################

    print("\nReordering for focus on CpG")

    # Initialise resulting CpG dictionary
    dic_allcpgs = {}
    for ele in df_allLat.index:
        dic_allcpgs[ele] = dict()

    ### fucking complicated loop to reparse with focus on CpGs
    for latFeature_key, _ in dic_allLatFeatures.items():
        for effect_key, _ in dic_allLatFeatures[latFeature_key].items(): 
            for cpg in dic_allLatFeatures[latFeature_key][effect_key].index:
                dic_allcpgs[cpg][f"latFeature_{latFeature_key}"] = effect_key


    ## encode pertubation effect to numerical values (to do some calculations)
    dic_encoding={"none":0, 
                "low":1, 
                "medium":2, 
                "high":3}

    df = pd.DataFrame.from_dict(dic_allcpgs, orient="index")
    df = df.replace(dic_encoding).T

    df.to_csv(f"results/{name}/perturbations/quantileCutoff_CpGfocus.csv")

    return dic_allLatFeatures


def applyStdThreshold(data, name, plot=False, save=False):
    '''
    Thresholds: 
        - High effect (of perturbation): Mean+3*STD and higher
        - Intermediate effect: between Mean+2*STD and Mean+3*STD
        - Low effect: between Mean+1*STD and Mean+2*STD
        - No effect: lower than Mean+1*STD 
    '''   

    c_high = sns.color_palette("Greens")[5]
    c_medium = sns.color_palette("Greens")[3]
    c_low = sns.color_palette("Greens")[1]
    c_none = sns.color_palette("Greens")[0]

    dic = {}
    mean = data.mean()
    std = data.std()

    thresh_high = mean+3*std
    cpgs_high = data[data > thresh_high]
    dic["high"] = cpgs_high

    thresh_medium = mean+2*std
    cpgs_medium = data[(data > thresh_medium) & (data < thresh_high)]
    dic["medium"] = cpgs_medium

    thresh_low = mean+1*std
    cpgs_low = data[(data > thresh_low) & (data < thresh_medium)]
    dic["low"] = cpgs_low   

    cpgs_none = data[data < thresh_low]
    dic["none"] = cpgs_none   
     
    if plot:
        fig, ax1 = plt.subplots(1,1, figsize=(8,6))
        sns.kdeplot(data=data, ax=ax1, color="black", linewidth=3)
        ax1.axvspan(thresh_high, data.max()+1 ,facecolor=c_high, alpha=0.8, label="high", lw=0) #alpha=0.25
        ax1.axvspan(thresh_medium, thresh_high ,facecolor=c_medium, label="medium",lw=0) #alpha=0.25
        ax1.axvspan(thresh_low, thresh_medium ,facecolor=c_low, label="low", lw=0)
        ax1.axvspan(-3, thresh_low,facecolor=c_none, label="none", lw=0)
        ax1.set_xlim(data.min()-data.mean()*.2, data.max()+data.max()*.05)
        ax1.legend()
        ax1.set_xlabel("Perturbation effect across all CpGs")
        ax1.set_title(f"Latent feature: {data.name}")
        plt.show()

    if save:
        resultsDir = f"results/{name}/perturbations/latFeature_{data.name}"
        os.makedirs(resultsDir, exist_ok=True)
        #print(f"Saved to: {resultsDir}")

        ### save CpGs in .txt file
        with open(os.path.join(resultsDir, f"cpgs_high.txt"), "w") as f:
            for ele in cpgs_high.index.tolist():
                f.write(ele+"\n")  
        with open(os.path.join(resultsDir, f"cpgs_medium.txt"), "w") as f:
            for ele in cpgs_medium.index.tolist():
                f.write(ele+"\n")  
        with open(os.path.join(resultsDir, f"cpgs_low.txt"), "w") as f:
            for ele in cpgs_low.index.tolist():
                f.write(ele+"\n")
        with open(os.path.join(resultsDir, f"cpgs_none.txt"), "w") as f:
            for ele in cpgs_none.index.tolist():
                f.write(ele+"\n")
        ##
        with open(os.path.join(resultsDir, f"cpgs_high_medium_low.txt"), "w") as f:
            for ele in cpgs_high.index.tolist():
                f.write(ele+"\n")
            for ele in cpgs_medium.index.tolist():
                f.write(ele+"\n")
            for ele in cpgs_low.index.tolist():
                f.write(ele+"\n")                                                      
         
    return dic    