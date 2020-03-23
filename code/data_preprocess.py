from rdkit import Chem
import numpy as np
import pickle
import ipdb

def process_NCI(filename):
    suppl = Chem.SDMolSupplier('../data/raw_datasets/NCI_balanced/{}.sdf'.format(filename), sanitize=False)
    mols = [x for x in suppl]
    dict_mol = {}
    for mol_idx, mol in enumerate(mols):
        dict_mol[mol_idx] = {}
        dict_mol[mol_idx]['adj'] = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
        dict_mol[mol_idx]['label'] = mol.GetProp('value')
        for atom in mol.GetAtoms():
            for neigh in atom.GetNeighbors():
                bond_idx=mol.GetBondBetweenAtoms(atom.GetIdx(),neigh.GetIdx()).GetIdx()
                dict_mol[mol_idx]['adj'][atom.GetIdx()][neigh.GetIdx()] = mol.GetBondWithIdx(bond_idx).GetBondType()
            dict_mol[mol_idx][atom.GetIdx()] = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetImplicitValence(),
                                                int(atom.GetIsAromatic())]
    pickle.dump(dict_mol, open('data/NCI/{}'.format(filename), 'wb'))


def process_PTC(filename):
    fboj = open('data/raw_datasets/PTC_pn/{}.smi'.format(filename))
    dict_mol = {}
    for mol_idx, eachline in enumerate(fboj):
        t = eachline.strip().split(',')
        dict_mol[mol_idx] = {}
        dict_mol[mol_idx]['label'] = int(t[1])
        mol = Chem.MolFromSmiles(t[2], sanitize=False)
        dict_mol[mol_idx]['adj'] = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
        for atom in mol.GetAtoms():
            for neigh in atom.GetNeighbors():
                dict_mol[mol_idx]['adj'][atom.GetIdx()][neigh.GetIdx()] = 1
            dict_mol[mol_idx][atom.GetIdx()] = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetImplicitValence(),
                                                int(atom.GetIsAromatic())]
    pickle.dump(dict_mol, open('data/PTC_pn/{}'.format(filename), 'wb'))

def format_file(task_type):
    if task_type == 'NCI':
        idx_list = [1, 33, 41, 47, 81, 83, 109, 123, 145]
    elif task_type == 'PTC_pn':
        idx_list = ['FM', 'FR', 'MM', 'MR']
    data = {}
    for data_idx in idx_list:
        if task_type == 'NCI':
            data[data_idx] = pickle.load(open('../data/{0}/{1}-balance'.format(task_type, data_idx), 'rb'))
        elif task_type == 'PTC_pn':
            data[data_idx] = pickle.load(open('../data/{0}/PTC_pn_{1}'.format(task_type, data_idx), 'rb'))

    element_dict = {}
    bond_dict = {}
    elem_cnt = 0
    for graphset in data:
        for graph in data[graphset]:
            for node in data[graphset][graph]:
                if node in ['adj', 'label']:
                    continue
                if not data[graphset][graph][node][0] in element_dict:
                    element_dict[data[graphset][graph][node][0]] = elem_cnt
                    elem_cnt += 1
                data[graphset][graph][node][0] = element_dict[data[graphset][graph][node][0]]
            unique_bond = np.unique(data[graphset][graph]['adj'])
            for bond in unique_bond:
                if bond not in bond_dict:
                    bond_dict[bond] = 1
        print(element_dict)
    print('element dict final is')
    print(element_dict)
    print(bond_dict)

    with open('../data/{0}/element_dict'.format(task_type, graphset), 'wb') as f:
        pickle.dump(element_dict, f)

    for graphset in data:
        with open('../data/{0}/{0}_{1}.txt'.format(task_type, graphset), 'w') as f:
            f.write('{0}\n'.format(len(data[graphset])))
            for graph in data[graphset]:
                f.write('{0}\t{1}\n'.format(len(data[graphset][graph])-2, int(float(data[graphset][graph]['label']))))
                for node_idx in range(len(data[graphset][graph])-2):
                    f.write('{0}\t'.format(data[graphset][graph][node_idx][0]))
                    neigh=np.nonzero(data[graphset][graph]['adj'][node_idx])[0]
                    m=neigh.shape[0]
                    f.write('{0}\t'.format(m))
                    for i in range(m):
                        f.write('{0}\t'.format(neigh[i]))
                    f.write('\t'.join([str(x) for x in data[graphset][graph][node_idx][1:]]))
                    f.write('\n')



if __name__ == '__main__':
    NCI_idx_list=[ 33, 41, 47, 81, 83, 109, 123, 145]
    for NCI_idx in NCI_idx_list:
        process_NCI('{}-balance'.format(NCI_idx))
    PTC_idx_list=['FM', 'FR', 'MM', 'MR']
    for PTC_idx in PTC_idx_list:
        process_PTC('PTC_pn_{}'.format(PTC_idx))
    format_file('NCI')
