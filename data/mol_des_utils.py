import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from scipy.stats import moment
def m2s(m): return Chem.MolToSmiles(m)
def s2m(s): return Chem.MolFromSmiles(s)
from scipy.spatial import ConvexHull
from rdkit.Chem import Descriptors
from tqdm import tqdm
from openbabel import pybel

pt = Chem.GetPeriodicTable()

def get_dist_matrix(pos):
    locs = pos
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat

def dist_bw2(p,c): return np.sqrt(sum((p-c)**2))
def dist_bw(pos,c): return np.sqrt(np.sum((pos - c)**2, axis=1))


def dist_bw(x, array):
    return np.sum((array - x)**2, axis=1)**.5

def get_xyz(path):
    f = open(path, 'r')
    f = f.readlines()
    atoms = f[2:]
    
    natoms = int(f[0].strip('\n'))
    
    
    atom_z = [atoms[i].split()[0] for i in range(natoms)]
    pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms)]
    
    return atom_z, np.array(pos)

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def convex_hull_volume_bis(pts):
    ch = ConvexHull(pts)
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex), ch.simplices))

    tets = ch.points[simplices]
    tvols  = tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3])
    tvols = list(tvols)

    volsum = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))
    volsum = float(volsum)
    return volsum, tvols


def dist_bw(x, array):
    return np.sum((array - x)**2, axis=1)**.5

def natoms_inlayers(pos):

    c = np.mean(pos, axis=0)
    dist_c = dist_bw(c, pos)
    d_sorted = np.sort(dist_c)

    n1 = len(np.where(d_sorted<3)[0])
    n2 = len(np.where(np.logical_and(d_sorted<5, d_sorted>=3))[0])
    n3 = len(np.where(np.logical_and(d_sorted<7, d_sorted>=5))[0])
    n4 = len(np.where(np.logical_and(d_sorted<9, d_sorted>=7))[0])
    n5 = len(np.where(np.logical_and(d_sorted<11, d_sorted>=9))[0])
    n6 = len(np.where(np.logical_and(d_sorted<13, d_sorted>=11))[0])
    
    return n1, n2, n3, n4, n5, n6


def get_nfrags(mol, most_common ):
    
    atoms = mol.GetAtoms()
    matches =[]
    for f in most_common:
        
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
        qp.adjustDegree=True
        qp.adjustDegreeFlags=Chem.ADJUST_IGNOREDUMMIES
        m = Chem.MolFromSmiles(f)
        qm = Chem.AdjustQueryProperties(m,qp)

        matches_frag = mol.GetSubstructMatches(qm)
        if len(matches_frag)!=0:

            matches.append([f, len(matches_frag)])
    return dict(matches)


def create_momentf(df, xyz_dir, id_column, most_common):
    df = df.copy(deep=True)

    for j in range(len(most_common)):
        df.loc[:,most_common[j]] = 0.0
        
    for i in tqdm(range(df.shape[0])):

        fname = df.loc[i, id_column]
        path = xyz_dir + fname + ".xyz"
        try:
            anum, pos = get_xyz(path)
        except:
            continue


        c = np.mean(pos, axis=0)

        dist_c = dist_bw(c, pos)
        max_c = np.argmax(dist_c)
        min_c = np.argmin(dist_c)

        dist_max = dist_bw(pos[max_c], pos)
        dist_min = dist_bw(pos[min_c], pos)

        for j in range(1,11):
            df.loc[i,"maxM"+str(j)] = moment(dist_max, moment=j)
            df.loc[i,"minM"+str(j)] = moment(dist_min, moment=j)
            df.loc[i,"cenM"+str(j)] = moment(dist_c, moment=j)
            
            
            
        n1, n2, n3, n4, n5, n6 = natoms_inlayers(pos)
        df.loc[i, "n1"] = n1
        df.loc[i, "n2"] = n2
        df.loc[i, "n3"] = n3
        df.loc[i, "n4"] = n4
        df.loc[i, "n5"] = n5
        df.loc[i, "n6"] = n6

        try:
            volume, _ = convex_hull_volume_bis(pos)
            df.loc[i, "vol"] = volume
        except:
            df.loc[i, "vol"] = 0
    
        smiles = df.loc[i,'smiles']
        mol = s2m( smiles )
        nf = get_nfrags(mol, most_common)

        keys = [j for j in nf.keys()]

        for key in keys:

            df.loc[i, key ] = nf[key]
        
    return df