from torch.utils.data import Dataset
import random
from utils import split_into_sentences
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
RDLogger.DisableLog('rdApp.*')


class smi_txt_dataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, unconditional=False, raw_description=False):
        self.data = []
        for dp in data_path:
            with open(dp, 'r') as r:
                self.data += [l.strip() for l in r.readlines()][1 if dp.endswith('.csv') else 0:]
        # if not raw_description: self.data = [l for l in self.data if 'natural product' not in l]

        if shuffle:
            random.shuffle(self.data)

        if data_length:
            self.data = self.data[:data_length]
        self.unconditional = unconditional
        self.raw_description = raw_description
        self.null_text = "no description."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            contents = self.data[index].split('\t')
            # print(len(contents), contents)
            if len(contents) == 3:
                cid, smiles, description = contents
            elif len(contents) == 2:
                smiles, description = contents
            elif len(contents) == 1:
                smiles, description = contents[0], self.null_text
            else:
                raise ValueError("Invalid data format in the dataset!")
            if self.unconditional:
                description = self.null_text

            # augment descriptions
            if not self.raw_description and description != self.null_text:
                description = sentence_randomize(description, only_one=True)
            else:
                pass

            if '.' in smiles:
                smiles = max(smiles.split('.'), key=len)
            mol = Chem.MolFromSmiles(smiles)
            sc_list = list(EnumerateStereoisomers(mol))
            mol = random.choice(sc_list)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

            return '[CLS]' + smiles, description
            # return (calculate_property(smiles) - self.p_mean) / self.p_std, description
        except Exception as e:
            print(e)
            print('aaa', self.data[index])
            raise NotImplementedError


class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, is_train=True):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()

        self.data = [l.strip() for l in lines]

        if shuffle:
            random.shuffle(self.data)

        self.train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data[index].split('\t')[0]
        smiles2 = smiles
        if random.random() > 0.:
            try:
                mol = Chem.MolFromSmiles(smiles)
                sc_list = list(EnumerateStereoisomers(mol))
                if self.train and len(sc_list) > 1:
                    mol, mol2 = random.sample(sc_list, k=2)
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    smiles2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=True)
                else:
                    mol = random.choice(sc_list)
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except:
                pass
        if self.train and smiles2 != smiles:
            return '[CLS]'+smiles+'Q[CLS]'+smiles2
        return '[CLS]' + smiles

        # smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        # return '[CLS]' + smiles


def sentence_randomize(description, only_one=False):
    desc = split_into_sentences(description)
    desc2, tmp = [], []
    for d in desc:
        if not d[0].isalpha():
            if tmp:
                tmp.append(d)
            else:
                if len(desc2) == 0:
                    desc2.append(d)
                    continue
                    # raise ValueError(d, d[0].isalpha(), d.isupper(), desc)
                head = desc2.pop()
                tmp.append(head)
                tmp.append(d)
        else:
            if tmp:
                desc2.append(' '.join(tmp))
                tmp = []
            desc2.append(d)
    if tmp:
        desc2.append(' '.join(tmp))
    # print(desc2)
    forced = random.randint(0, len(desc2) - 1)
    # forced = 0
    if only_one:
        desc2 = [desc2[forced]]
    else:
        desc2 = [d for i, d in enumerate(desc2) if random.random() < 0.5 or i == 0]
    description = ' '.join(desc2)
    return description
