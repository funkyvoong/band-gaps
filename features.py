import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import mendeleev

df = pd.read_csv('CODids.csv',encoding = 'unicode_escape')
df2= pd.read_csv('bandgaps.csv',encoding = 'unicode_escape')
df['bandgaps']=df2['bgs']
df=df.loc[12000:, :]


featurelist= ['_chemical_formula_sum', "_chemical_formula_weight",  "_space_group_IT_number",
"_symmetry_cell_setting",  "_symmetry_space_group_name_H-M",  "_cell_angle_alpha",  "_cell_angle_beta",
"_cell_angle_gamma",  "_cell_formula_units_Z",  "_cell_length_a",  "_cell_length_b",  "_cell_length_c",
"_cell_measurement_reflns_used",  "_cell_measurement_temperature",  "_cell_measurement_theta_max",
"_cell_measurement_theta_min",  "_cell_volume",  "_diffrn_radiation_wavelength",
"_diffrn_reflns_av_R_equivalents",  "_diffrn_reflns_av_sigmaI/netI",  "_diffrn_reflns_limit_h_max",
"_diffrn_reflns_limit_h_min",  "_diffrn_reflns_limit_k_max",  "_diffrn_reflns_limit_k_min",
"_diffrn_reflns_limit_l_max",  "_diffrn_reflns_limit_l_min",  "_diffrn_reflns_number",
"_diffrn_reflns_theta_full",  "_diffrn_reflns_theta_max",  "_diffrn_reflns_theta_min",
"_exptl_absorpt_coefficient_mu",  "_exptl_absorpt_correction_T_max",  "_exptl_absorpt_correction_T_min"]


# choosing features
#featurelist = ['_chemical_formula_sum','_chemical_formula_weight',"_symmetry_space_group_name_H-M"]
for x in featurelist:
    stringlist=['_chemical_formula_sum', "_symmetry_space_group_name_H-M"]
    if x in stringlist:
        df[x]= ''
    else:
        df[x]=-100.1


# get info
print("accessing database")
for x in df['ids']:
    id= str(x)
    index=df.loc[df['ids'] == x].index[0]
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

print("encoding labels")
# encoding space group
space_vals = df["_symmetry_space_group_name_H-M"]
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(space_vals)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
space_vals_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded[0])
#print(len(onehot_encoded[0]))

# encoding symmetry cell setting
cell_setting = df["_symmetry_cell_setting"]
#print(set(cell_setting))
cell_label = LabelEncoder()
cell_encoded = cell_label.fit_transform(cell_setting)
cell_onehot = OneHotEncoder(sparse=False)
cell_encoded = cell_encoded.reshape(len(cell_encoded), 1)
cell_vals_encoded = onehot_encoder.fit_transform(cell_encoded)
#print(cell_vals_encoded[0])
#print(len(cell_vals_encoded[0]))

def corr_calc(x, y):
    from scipy.stats import pearsonr, spearmanr
    covariance = np.cov(x,y)
    pcorr, _ = pearsonr(x,y)
    scorr, _ = spearmanr(x,y)
    print('Covariance: ')
    print(covariance)
    print('Pearsons Correlation: ', pcorr)
    print('Spearmans Correlation: ', scorr)

#corr_calc(df['_chemical_formula_weight'],df['bandgaps'])

#load xyz
xyz_df = pd.read_excel('xyz2.xlsx', encoding = 'unicode_escape')

molecules = []
mi = []

for index, row in xyz_df.itertuples():
    try:
        if 'Lattice' in row:
            pass
        else:
            line = row.split()
            mi.append(line)
    except TypeError:
        molecules.append(mi)
        mi = []
        pass
print("computing coulombs")
# first 1000 molecules xyz
molec = molecules[5715:]
#print(molec[:][:][0])
size_molecs = [len(m) for m in molec]

max_molec = 208 #208 for omdb
df['num_atoms']=size_molecs
#print(len(df2['bgs']))
def get_coulombmat(molecule):
    import mendeleev
    """
    takes a molecule as input, each row has molecule name and xyz coordinates
    returns coulomb matrix
    """
    atoms = []
    xyzmatrix = []
    num_atoms = len(molecule)
    #print(num_atoms)
    for line in molecule:
        elem = mendeleev.element(line[0])
        atoms.append(elem)
        cij = np.zeros((num_atoms, num_atoms))
        xyzmatrix.append([[float(line[1]), float(line[2]), float(line[3])]])
    #xyzmatrix = [[atom.position.x, atom.position.y, atom.position.z] for atom in molecule.atoms]

    chargearray = [a.atomic_number for a in atoms]
    #print(chargearray)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                cij[i][j] = 0.5 * chargearray[i] ** 2.4  # Diagonal term described by Potential energy of isolated atom
            else:
                dist = np.linalg.norm(np.array(xyzmatrix[i]) - np.array(xyzmatrix[j]))
                cij[i][j] = chargearray[i] * chargearray[j] / dist  # Pair-wise repulsion
    return cij

def pad_cmat(cmat, ref_shape=208):
    result = np.zeros((ref_shape, ref_shape))
    result[:cmat.shape[0],:cmat.shape[1]] = cmat
    return result

#c = get_coulombmat(molec[0])
#padded = pad_cmat(c, max_molec)#must run pad_cmat on each molecule 1 at a time

df['coulomb_original']=[get_coulombmat(m) for m in molec]
df['coulomb_padded']=df['coulomb_original'].apply(pad_cmat)

# sorted eigenvalues of coulomb matrix
df['eig']=df['coulomb_padded'].apply(np.linalg.eigvals)
df['eig']=df['eig'].apply(sorted, reverse=True)

#sorted_eig = sorted(np.linalg.eigvals(padded), reverse=True)

#print to read_csv
exported= df.to_csv(r'./cifdata12500.csv', index= None, header=True)
