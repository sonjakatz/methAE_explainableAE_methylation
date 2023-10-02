import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")


def loadData():
    df_manifest = pd.read_csv(f"{PATH_data}/humanmethylation450_15017482_v1-2.csv", skiprows=7, low_memory=False, header=0, index_col="IlmnID")
    #df_manifest = df_manifest[df_manifest["CHR"] == "22"]
    return df_manifest

def hypergeometricTest(M, n, N, x):

    '''
    Hypergeometric test

    Population size (N): total number of associations  --> e.g. all "regulatory_feature_group*" = 170023
    Number of "successes" in population: total number of associations of interest --> e.g "promotor associated*" = 91954
    Sample size (n): "number of draws" --> e.g. 24

    ### *df_random["Regulatory_Feature_Group"].value_counts()

    '''

    from scipy.stats import hypergeom

    # print("\n\n\tHYPERGEOMETRIC TEST")
    # print(f"Population size: {M}")
    # print(f"Number successes in Population: {n}")
    # print(f"Sample size: {N}")
    # print(f"Number of drawn successes: {x}")

    pval = hypergeom.sf(x-1, M, n, N)
    #print(f"\n\tHypergeometric test \t p-value calculated: {pval}")

    return pval

def locationOnChromosome(df):
    '''
    '''
    ### Plot
    fig, ax1 = plt.subplots(figsize=(30,5))
    ax1.bar((df["MAPINFO"]/1000).to_list(), height=1,  width=50)
    ax1.set_xlim(0,50000)
    ax1.set_xlabel("Coordinates")
    fig.tight_layout()
    plt.savefig(f"{PATH_figs}/location.png")
    print(f"\n##################### LOCATION ON CHROMOSOME #####################")
    print(f"...generated figure {PATH_figs}/location.png")
    return 

def locationRelativeGene(df, df_manifest):
    '''
    UCSC_RefGene_Group: "Gene region feature category describing the CpG position"

    - multiple mappings possible due to splicing variants: take only most common gene region into account
    '''

    dic_RefGene_Group = {}
    for i in df["UCSC_RefGene_Group"].to_list():
        if type(i) == float:
            i = str(i)    
        if type(i) == str:
            i = i.split(";")
            ct = Counter(i)   # count items in list 
            i = ct.most_common()[0][0]   # assign most common splicing variant to gene region
        if i not in dic_RefGene_Group.keys():
            # initialise key in dic
            dic_RefGene_Group[i] = 1
        else:
            # add to count in dic
            dic_RefGene_Group[i] += 1      

    ### Plot ###
#     fig, ax2 = plt.subplots(figsize=figsize_square)
#     ax2.bar(dic_RefGene_Group.keys(), dic_RefGene_Group.values())
#     ax2.set_xticklabels(labels=dic_RefGene_Group.keys(),rotation=90)
#     ax2.set_title("Gene region feature category describing the CpG position")
#     fig.tight_layout()
    #plt.savefig(f"{PATH_figs}/geneRegion.png")
    #print(f"\nLocation relative to gene ... \t generated figure {PATH_figs}/geneRegion.png")

    ### Hypergeometric test ###
    df_hgt = pd.DataFrame()
    for target in dic_RefGene_Group.keys():
        if target != "nan":
            M = df_manifest["UCSC_RefGene_Group"].value_counts().sum() 
            n = df_manifest["UCSC_RefGene_Group"].value_counts()[target] 
            N = df.shape[0]
            x = dic_RefGene_Group[target]
            pval = hypergeometricTest(M, n, N, x)
            df_hgt.loc[target, "count"] = f"{x}/{N}"
            df_hgt.loc[target, "pval"] = pval
            df_hgt.loc[target, "context"] = "relativeGene"

    print(f"\n##################### LOCATION RELATIVE TO GENE #####################")
    print(f"...generated figure {PATH_figs}/location.png; generated test report")
    print(df_hgt[df_hgt["pval"] < 0.05])

    return df_hgt

def associationIsland(df):
    '''
    UCSC_CpG_Islands_Name: "Chromosomal coordinates of the CpG Island from UCSC."
    '''

    dic_Island = {}
    for i in df["UCSC_CpG_Islands_Name"].to_list():
        if i not in dic_Island.keys():
            # initialise key in dic
            dic_Island[i] = 1
        else:
                # add to count in dic
            dic_Island[i] += 1

#     ### Plot ###
#     fig, ax3 = plt.subplots(figsize=figsize_horizontal)
#     ax3.bar(list(map(str,dic_Island.keys())), dic_Island.values())
#     ax3.set_xticklabels(labels=list(map(str,dic_Island.keys())),rotation=90)
#     ax3.set_title("Chromosomal coordinates of the CpG Island")
#     fig.tight_layout()
#     #plt.savefig(f"{PATH_figs}/associationIsland.png")
#     print(f"\n##################### ASSOCIATION TO ISLAND #####################")
#     print(f"generated figure {PATH_figs}/associationIsland.png")

    return

def locationRelativeIsland(df, df_manifest):
    '''
    Relation_to_UCSC_CpG_Island: "The location of the CpG relative to the CpG island"
    '''
    dic_Relation_Island = {}
    for i in df["Relation_to_UCSC_CpG_Island"].to_list():
        if i not in dic_Relation_Island.keys():
            # initialise key in dic
            dic_Relation_Island[i] = 1
        else:
                # add to count in dic
            dic_Relation_Island[i] += 1

#     ### Plot ###
#     fig, ax4 = plt.subplots(figsize=figsize_square)
#     ax4.bar(list(map(str,dic_Relation_Island.keys())), dic_Relation_Island.values())
#     ax4.set_xticklabels(labels=list(map(str,dic_Relation_Island.keys())),rotation=90)
#     ax4.set_title("The location of the CpG relative to the CpG island")
#     fig.tight_layout()
#     #plt.savefig(f"{PATH_figs}/relationIsland.png")
#     #print(f"\nLocation relative to island ... \t generated figure {PATH_figs}/relationIsland.png")

    ### Hypergeometric test ###
    df_hgt = pd.DataFrame()
    for target in dic_Relation_Island.keys():
        if target is not np.nan:
            M = df_manifest["Relation_to_UCSC_CpG_Island"].value_counts().sum() 
            n = df_manifest["Relation_to_UCSC_CpG_Island"].value_counts()[target] 
            N = df.shape[0]
            x = dic_Relation_Island[target]
            pval = hypergeometricTest(M, n, N, x)
            df_hgt.loc[target, "count"] = f"{x}/{N}"
            df_hgt.loc[target, "pval"] = pval
            df_hgt.loc[target, "context"] = "relativeIsland"

    print(f"\n##################### LOCATION RELATIVE TO ISLAND #####################")
    print(f"...Location relative to island ... \t generated figure {PATH_figs}/relationIsland.png; generated test report")
    print(df_hgt[df_hgt["pval"] < 0.05])
    return df_hgt

def associationRegulatoryFeature(df, df_manifest):
    '''
    Regulatory_Feature_Group: 
    '''
    dic_RegulatoryFeature = {}
    for i in df["Regulatory_Feature_Group"].to_list():
        if i not in dic_RegulatoryFeature.keys():
            # initialise key in dic
            dic_RegulatoryFeature[i] = 1
        else:
                # add to count in dic
            dic_RegulatoryFeature[i] += 1


#     fig, ax5 = plt.subplots(figsize=figsize_square)
#     ax5.bar(list(map(str,dic_RegulatoryFeature.keys())), dic_RegulatoryFeature.values())
#     ax5.set_xticklabels(labels=list(map(str,dic_RegulatoryFeature.keys())),rotation=90)
#     ax5.set_title("Regulatory Feature")
#     fig.tight_layout()
#     #plt.savefig(f"{PATH_figs}/regulatoryFeature.png")
#     #print(f"\nRegulatory feature group ... \t generated figure {PATH_figs}/regulatoryFeature.png")

    ### Hypergeometric test ###
    df_hgt = pd.DataFrame()
    for target in dic_RegulatoryFeature.keys():
        if target is not np.nan:
            M = df_manifest["Regulatory_Feature_Group"].value_counts().sum() 
            n = df_manifest["Regulatory_Feature_Group"].value_counts()[target] 
            N = df.shape[0]
            x = dic_RegulatoryFeature[target]
            pval = hypergeometricTest(M, n, N, x)
            df_hgt.loc[target, "count"] = f"{x}/{N}"
            df_hgt.loc[target, "pval"] = pval
            df_hgt.loc[target, "context"] = "regulatoryFeature"
    print(f"\n##################### REGULATORY FEATURE GROUP #####################")
    print(f"Regulatory feature group ... \t generated figure {PATH_figs}/regulatoryFeature.png; generated test report")
    print(df_hgt[df_hgt["pval"] < 0.05])
    return df_hgt

def dhsSite(df, df_manifest):
    '''
    DNase I Hypersensitivity Site (experimentally determined by the ENCODE project).
    '''

    dic_dhs = {}
    for i in df["DHS"].to_list():
        if i not in dic_dhs.keys():
            # initialise key in dic
            dic_dhs[i] = 1
        else:
                # add to count in dic
            dic_dhs[i] += 1

    ## rename key name from np.nan to False
    dic_dhs[False] = dic_dhs.pop(np.nan)
    ## rename np.nan in manifest to False
    df_manifest["DHS"][df_manifest["DHS"].isna()] = False

    df_hgt = pd.DataFrame()
    for target in dic_dhs.keys():
        M = df_manifest["DHS"].value_counts().sum() 
        n = df_manifest["DHS"].value_counts()[target] 
        N = df.shape[0]
        x = dic_dhs[target]
        pval = hypergeometricTest(M, n, N, x)
        
        df_hgt.loc[str(target), "count"] = f"{x}/{N}"
        df_hgt.loc[str(target), "pval"] = pval
        df_hgt.loc[str(target), "context"] = "DHS"


    print(f"\n##################### DHS SITE #####################")
    print(f"DHS site ... \t; generated test report")
    print(df_hgt[df_hgt["pval"] < 0.05])    
    return df_hgt

def enhancerSite(df, df_manifest):
    '''
    '''

    dic_enhancer = {}
    for i in df["Enhancer"].to_list():
        if i not in dic_enhancer.keys():
            # initialise key in dic
            dic_enhancer[i] = 1
        else:
                # add to count in dic
            dic_enhancer[i] += 1

    ## rename key name from np.nan to False
    dic_enhancer[False] = dic_enhancer.pop(np.nan)
    ## rename np.nan in manifest to False
    df_manifest["Enhancer"][df_manifest["Enhancer"].isna()] = False

    df_hgt = pd.DataFrame()
    for target in dic_enhancer.keys():
        M = df_manifest["Enhancer"].value_counts().sum() 
        n = df_manifest["Enhancer"].value_counts()[target] 
        N = df.shape[0]
        x = dic_enhancer[target]
        pval = hypergeometricTest(M, n, N, x)
        
        df_hgt.loc[str(target), "count"] = f"{x}/{N}"
        df_hgt.loc[str(target), "pval"] = pval
        df_hgt.loc[str(target), "context"] = "Enhancer"

    print(f"\n##################### ENHANCER #####################")
    print(f"Enhancer site ... \t; generated test report")
    print(df_hgt[df_hgt["pval"] < 0.05])  

    return df_hgt

def associationGenomic(df):
    '''
    Probe_SNPs: 
    '''    
    ### SNPs
    dic_snp = {}
    for i in df["Probe_SNPs"].to_list():
        if i not in dic_snp.keys():
            # initialise key in dic
            dic_snp[i] = 1
        else:
                # add to count in dic
            dic_snp[i] += 1

#     fig, ax6 = plt.subplots(figsize=figsize_square)
#     ax6.bar(list(map(str,dic_snp.keys())), dic_snp.values())
#     ax6.set_xticklabels(labels=list(map(str,dic_snp.keys())),rotation=90)
#     ax6.set_title("SNPs")
#     fig.tight_layout()
#     #plt.savefig(f"{PATH_figs}/SNP.png")
#     print(f"\nAssociated SNPs ... \t generated figure {PATH_figs}/SNP.png\n")

    ### meQTL
    # Load cis / trans mappings
    df_qtl_cis = pd.read_csv(f"{PATH_data}/cisPe6_chr22_clean.csv-2e11.csv", index_col="CpG")
    df_qtl_trans = pd.read_csv(f"{PATH_data}/trans_mQTLs_exclintra5Mb.csv", index_col="CpG")

    try:
        print(df_qtl_cis.loc[df.index,:])
    except KeyError:
        print("No matches for cis meQTL :(")

    try:
        print(df_qtl_trans.loc[df.index,:])
    except KeyError:
        print("No matches for trans meQTL :(")  

    return

def associationCellType(df):
    '''
    '''    
    df_celltypemarkers = pd.read_csv(f"{PATH_data}/cell_type_markers_cpgs.csv", index_col="IlmnID")

    try:
        print(df_celltypemarkers.loc[df.index,:])
    except KeyError:
        print("\nNo matches for cell-type markers :(")
    return    

def associationBiologicalPathway():
    '''
    '''
    print(f"\n##################### GO - KEGG TERMS #####################")
    if "GO_association.csv" in os.listdir(PATH_results):
        df_GO = pd.read_csv(f"{PATH_results}/GO_association.csv")
        df_GO_sig = df_GO[df_GO["FDR"] < 0.05]
        df_KEGG = pd.read_csv(f"{PATH_results}/KEGG_association.csv")
        df_KEGG_sig = df_KEGG[df_KEGG["FDR"] < 0.05]  

        print(f"\nNumber of GO terms significantly associated: {df_GO_sig.shape[0]}")
        print(f"Number of KEGG pathways significantly associated: {df_KEGG_sig.shape[0]}\n") 
        
        if df_GO_sig.shape[0] > 0:
            print(df_GO_sig)
        if df_KEGG_sig.shape[0] > 0:
            print(df_KEGG_sig)
        return df_GO_sig, df_KEGG_sig

    else:
        print("\nATTENTION: Run R-script for Pathway analysis first!\n")
        return pd.DataFrame(), pd.DataFrame()



def main():
    # Load CpG list
    cpglist =  pd.read_csv(os.path.join(args.cpgfile), header=None)[0].tolist()
    # Load association information (Illumina Manifest, ...)
    df_manifest = loadData()
    # Parse only relevant CpGs from Illumina manifest
    df = df_manifest.loc[cpglist,:]

    ### Run associations
    locationOnChromosome(df)

    df_relGene = locationRelativeGene(df, df_manifest)
    df_regFeature = associationRegulatoryFeature(df, df_manifest)

    associationIsland(df)
    df_relIsland = locationRelativeIsland(df, df_manifest)

    df_dhs = dhsSite(df, df_manifest)
    df_enhancer = enhancerSite(df, df_manifest)

    # associationGenomic(df)
    # associationCellType(df)
    goAssoc, keggAssoc = associationBiologicalPathway()

    df_summary = pd.concat([df_relGene, df_regFeature, df_relIsland, df_dhs, df_enhancer, goAssoc, keggAssoc])
    df_summary.to_csv(f"{PATH_results}/AssociationSummary.csv", index_label="index")
    return


if __name__ == '__main__':

    # general settings
    figsize_square = (7,7)
    figsize_horizontal = (20,7)

    # ArgParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpgfile")
    args = parser.parse_args()

    # PATHs
    PATH_data = "/data/scratch/skatz/PROJECTS/methylnet"
    PATH_results = "/".join(str.split(args.cpgfile, ".txt")[:-1])  # same output directory where CpG file lies
    os.makedirs(PATH_results, exist_ok=True)
    PATH_figs = f"{PATH_results}/figs"
    os.makedirs(PATH_figs, exist_ok=True)

    print(f"\nResults saved in: {PATH_results}")

    main()