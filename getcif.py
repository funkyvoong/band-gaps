import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

df = pd.read_csv('CODids.csv',encoding = 'unicode_escape')
df2= pd.read_csv('bandgaps.csv',encoding = 'unicode_escape')
df['bandgaps']=df2['bgs']
df=df.head(1000)

featurelist= ['_chemical_formula_sum', "_chemical_formula_weight",  "_space_group_IT_number",  "_symmetry_cell_setting",  "_symmetry_space_group_name_H-M",  "_cell_angle_alpha",  "_cell_angle_beta",  "_cell_angle_gamma",  "_cell_formula_units_Z",  "_cell_length_a",  "_cell_length_b",  "_cell_length_c",  "_cell_measurement_reflns_used",  "_cell_measurement_temperature",  "_cell_measurement_theta_max",  "_cell_measurement_theta_min",  "_cell_volume",  "_diffrn_radiation_wavelength",  "_diffrn_reflns_av_R_equivalents",  "_diffrn_reflns_av_sigmaI/netI",  "_diffrn_reflns_limit_h_max",  "_diffrn_reflns_limit_h_min",  "_diffrn_reflns_limit_k_max",  "_diffrn_reflns_limit_k_min",  "_diffrn_reflns_limit_l_max",  "_diffrn_reflns_limit_l_min",  "_diffrn_reflns_number",  "_diffrn_reflns_theta_full",  "_diffrn_reflns_theta_max",  "_diffrn_reflns_theta_min",  "_exptl_absorpt_coefficient_mu",  "_exptl_absorpt_correction_T_max",  "_exptl_absorpt_correction_T_min"]
for x in featurelist:
    stringlist=['_chemical_formula_sum', "_symmetry_space_group_name_H-M"]
    if x in stringlist:
        df[x]= ''
    else:
        df[x]=-100.1


for x in df['ids']:
    id= str(x)
    index=df.loc[df['ids'] == x].index[0]
    #print(index)
    url=requests.get('http://www.crystallography.net/cod/'+id+'.cif')
    htmltext = url.text
    for y in featurelist:
        matched_lines = [line for line in htmltext.split('\n') if y in line]
        if len(matched_lines)>0:
            if y=="_chemical_formula_sum" or y=="_symmetry_space_group_name_H-M":
                matched_lines= matched_lines[0].split("'")
                df.set_value(index, y, matched_lines[1])
            else:
                matched_lines= matched_lines[0].split(" ")
                k=matched_lines[-1]
                if '(' in k:
                    k= k.split('(')[0]
                if y!="_symmetry_cell_setting":
                    try:
                        k=float(k)
                    except ValueError:
                        k=-100
                if y=="_symmetry_cell_setting":
                    if k== "monoclinic":
                        k=1.0
                    elif k=="triclinic":
                        k=3.0
                    elif k=="orthorhombic":
                        k=5.0
                    else:
                        k=7.0
                df.set_value(index, y, k)

# f = plt.figure(figsize=(19, 15))
# plt.matshow(df.corr(), fignum=f.number)
# plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=45)
# plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=10)
# #plt.title('Correlation Matrix', fontsize=16);
# plt.show()


X_tsne = TSNE(learning_rate=100).fit_transform(df[df.columns.difference(['bandgaps','_chemical_formula_sum', "_symmetry_space_group_name_H-M"])])
X_pca = PCA().fit_transform(df[df.columns.difference(['bandgaps','_chemical_formula_sum', "_symmetry_space_group_name_H-M"])])

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['bandgaps'])
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['bandgaps'])
plt.show()
