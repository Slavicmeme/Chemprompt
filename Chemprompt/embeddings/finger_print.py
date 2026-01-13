from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np

class FingerprintGenerator:
    """
    A class to convert a list of SMILES strings into various molecular fingerprints.
    """

    SUPPORTED_FINGERPRINTS = ["Morgan", "MACCS", "RDKit", "AtomPair", "TopologicalTorsion", "Pattern"]

    def __init__(self, fp_type="Morgan", radius=2, n_bits=1024):
        """
        :param fp_type: Fingerprint type ("Morgan", "MACCS", "RDKit", "AtomPair", "TopologicalTorsion", "Pattern")
        :param radius: Radius for Morgan fingerprint (not applicable to MACCS, RDKit, etc.)
        :param n_bits: Bit size for fingerprints (MACCS is fixed at 166)
        """
        if fp_type not in self.SUPPORTED_FINGERPRINTS:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_mol(self, smiles):
        """Convert a SMILES string into an RDKit molecule object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol

    def get_fingerprint(self, smiles):
        """Generate a fingerprint from a SMILES string based on the selected type."""
        mol = self.smiles_to_mol(smiles)

        if self.fp_type == "Morgan":
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits))

        elif self.fp_type == "MACCS":
            return np.array(MACCSkeys.GenMACCSKeys(mol))

        elif self.fp_type == "RDKit":
            return np.array(Chem.RDKFingerprint(mol))

        elif self.fp_type == "AtomPair":
            return np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.n_bits))

        elif self.fp_type == "TopologicalTorsion":
            return np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=self.n_bits))

        elif self.fp_type == "Pattern":
            return np.array(Chem.PatternFingerprint(mol))

    def transform(self, smiles_list):
        """
        Convert a list of SMILES strings into fingerprint vectors.
        
        :param smiles_list: List of SMILES strings
        :return: Numpy array of fingerprints
        """
        fingerprints = []
        for smiles in smiles_list:
            try:
                fp = self.get_fingerprint(smiles)
                fingerprints.append(fp)
            except ValueError as e:
                print(f"Error processing SMILES: {smiles} -> {e}")
                fingerprints.append(None)

        return np.array([fp for fp in fingerprints if fp is not None])
