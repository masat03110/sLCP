# 🔬 Efficient Approximation Algorithms for Order-Dependent Protein Substructure Alignment


[![License: MIT](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/masat03110/sLCP/blob/main/LICENSE)
![Language](https://img.shields.io/badge/Language-C++%2FPython-blue)
[![Dataset](https://img.shields.io/badge/Dataset-PDB-success)](https://www.rcsb.org/)
![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)



## Overview

Detecting common substructures in 3D protein structures is a cornerstone of **structural bioinformatics** and plays a critical role in **drug discovery**. This repository introduces **two novel approximation algorithms** for solving the **sequential Largest Common Point-set (sLCP) problem** under the **bottleneck distance**—a computational geometry problem known for its computational hardness with direct applications to protein alignment.

We improve upon the state-of-the-art by dramatically reducing computational complexity while maintaining or improving accuracy. Our algorithms are designed to be **both practical and scalable**, handling large biomolecular structures with **real-world speed** and **biological precision**.

---

## 🚀 Key Contributions

- 📈 **Two new approximation algorithms** for order-dependent LCP under bottleneck distance  
  - Algorithm 1 (**AlignFastLIS**): Optimized version of a known method with reduced time complexity **O(n⁷ log n)**  
  - Algorithm 2 (**AlignFastLCS**): Achieves greater speed improvements under realistic datasets
- ⚡ **Up to several orders of magnitude speedup** compared to prior algorithms  
- 📊 **Empirical validation** on real datasets from the **Protein Data Bank (PDB)**  
- ✅ **Outperforms** existing methods in **both runtime and alignment quality**

---

## 📂 Repository Structure

```bash
.
├── src/         # Core algorithm implementations (C++ with Python bindings)
├── include/     # Header files for C++ implementation
├── samples/     # Sample PDB files used for testing
├── results/     # Output: alignment results and performance plots
├── README.md    # Project overview and instructions (this file)
├── Makefile     # Makefile
└── LICENSE      # License information (MIT)
```
---

## 🧬 Applications

- Comparative protein structure analysis  
- Functional motif detection  
- Drug target screening  

---

## 📦 Getting Started

### Requirements

- C++20 or later  
- Python 3.12+  
- [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page) – for linear algebra operations in C++  
- [argparse](https://github.com/p-ranav/argparse) – for command-line argument parsing in C++


### Installation

```bash
# Clone the repository
git clone https://github.com/masat03110/sLCP.git
cd sLCP

# (Optional) Download Eigen and argparse if not already installed
# You can place them in 'include/' or adjust the include path accordingly.

# Compile the main program
make
```

### Example Usage
```bash
# Align two PDB files using AlignFastLIS
./aligner samples/1pyv_trimmed.pdb samples/7my8_trimmed.pdb --Algorithm AlignFastLIS

# Visualize output using Python
python src/plot.py
python -m http.server 8000
```

For more detailed options and usage instructions, run:
```bash
./aligner -h
python src/plot.py -h
```


---

## 📈 Experimental Results

Our methods were benchmarked on multiple PDB datasets. Compared to previous exact and approximate solutions:

- ⏱️ **Speed**: Up to **1000× faster**  
- 🎯 **Accuracy**: Maintains or improves alignment fidelity  
- 📦 **Scalability**: Handles proteins with up to around a hundred residues

---

## 📝 Citation

If you use this work in your research, please cite:

```bash
@inproceedings{tsukahara2025lcp,
  title={Efficient and Accurate Approximation Algorithms for Protein Substructure Alignment},
  author={Tsukahara, Masahito},
  booktitle={Proceedings of [Conference Name] (coming soon)},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or submit an issue for discussion.

---

## 📫 Contact

For questions or collaborations:  
**Masahito Tsukahara**  
Email: tsuka03@hgc.jp

---

## ⭐ Acknowledgments

Special thanks to the open-source community and PDB for providing invaluable datasets.
