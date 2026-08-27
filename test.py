try:
    from ga_MEPmodel_bio import ga_MEPmodel_bio
    ga_MEPmodel_bio(8,1,[],1)
except:
    print("Biological model failed")

try:
    from ga_MEPmodel_bio import ga_MEPmodel_bio
    ga_MEPmodel_bio(8,0,[],1)
except:
    print("Biological NO RC model failed")

try:
    from ga_MEPmodel_pheno import ga_MEPmodel_pheno
    ga_MEPmodel_pheno(8,1,[],0)
except:
    print("Pheno model failed")
# from load_muap import load_muap
# load_muap(1, 0)