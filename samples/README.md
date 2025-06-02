# PDB File Examples for Algorithm Testing

This directory contains a small collection of **original and trimmed PDB files**, used for developing and testing structural bioinformatics algorithms.  
Each `*_trimmed.pdb` file is a simplified version of its corresponding original structure, retaining only selected atoms or models.

## 📁 File Structure

```
samples/
    1pyv.pdb # Original structure from RCSB PDB
    1pyv_trimmed.pdb # Partial version (first 10 C-alpha atoms)
    7my8.pdb # Original structure from RCSB PDB
    7my8_trimmed.pdb # Partial version (first 10 C-alpha atoms)
```

---

## 📄 Source Information and Citations

### 🔬 `1pyv.pdb`

- **Title**: *NMR solution structure of the mitochondrial F1b presequence peptide from Nicotiana plumbaginifolia*  
- **PDB ID**: [1PYV](https://www.rcsb.org/structure/1PYV)  
- **PDB DOI**: [10.2210/pdb1PYV/pdb](https://doi.org/10.2210/pdb1PYV/pdb)
- **Entry authors**: Moberg, P., Nilsson, S., Stahl, A., Eriksson, A.C., Glaser, E., Maler, L.

`1pyv_trimmed.pdb` contains **the first 10 C-alpha (CA) atoms** from the original structure. This fragment is useful for testing simplified structural analysis workflows

---

### 🔬 `7my8.pdb`

- **Title**: Fusion Peptide of SARS-CoV-2 Spike Rearranges into a Wedge Inserted in Bilayered Micelles
- **PDB ID**: [7MY8](https://www.rcsb.org/structure/7MY8)   
- **PDB DOI**: [10.2210/pdb7my8/pdb](https://doi.org/10.2210/pdb7my8/pdb)
- **Entry authors**: Koppisetti, R.K., Fulcher, Y.G., Van Doren, S.R.


`7my8_trimmed.pdb` contains **the first 10 C-alpha (CA) atoms** from the original structure, extracted for algorithm testing purposes.

---

## ⚠️ License and Usage

The original PDB files are sourced from the [RCSB Protein Data Bank](https://www.rcsb.org/) and are in the public domain.  
However, **citation of the original authors and structures is strongly recommended** when using or redistributing derivative data.

The trimmed files are manually extracted versions intended for **non-commercial research and educational use**.

---

## 🧪 Purpose

These files are provided to:

- Simplify algorithm development and testing
- Offer lightweight samples for debugging or demo purposes
- Enable reproducible and minimal test cases

---