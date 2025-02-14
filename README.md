# **Prokaryotic Promoter Prediction**  

## **Project Overview**  
This project focuses on enhancing the prediction of *Escherichia coli* promoters by utilizing transfer learning techniques with Nucleotide Transformer models.  

I employed pre-trained Nucleotide Transformer models with different configurations and datasets, which were fine-tuned using specific datasets from the **Prokaryotic Promoter Database (PPD)**. Further experiments were conducted to improve model accuracy by adjusting training steps and the **Low-Rank Adaptation (LoRA)** dropout rate.  

The models were evaluated against a dataset comprising **865 natural, experimentally validated *E. coli* K-12 promoter sequences** and **1,000 randomly generated negative sequences**.  

---

## **Table of Contents**  
- [Installation & Scripts](#installation--scripts)  
- [Data](#data)  
- [Findings](#findings)  
- [Citations](#citations)  

---

## **Installation & Scripts**  
The experiments were conducted using **Python notebooks on Google Colab**. Figures were visualized using the following script:  

 `Bioinformatics_Project/figures/figures.py`  

---

## **Data**  
The main data files required to run the experiments include:  
  
**Training Data:**  
- `Bioinformatics_Project/PPD/train_data/train.csv`  
- `Bioinformatics_Project/PPD/train_data/train_coli.csv`  

**Testing Data:**  
- `Bioinformatics_Project/PPD/test_data/test.csv`  

---

## **Findings**  
The results and findings of this project are documented in:  
- `02-604_Bioinformatics_Project_FinalReport.pdf`  

---

## **Citations**   
1. J. Wright, *Gene Control*. Scientific e-Resources, 2019.  
2. M. E. Consens *et al.*, “To Transformers and Beyond: Large Language Models for the Genome,” *arXiv preprint arXiv:2311.07621*, 2023.  
3. M. H. A. Cassiano and R. Silva-Rocha, “Benchmarking bacterial promoter prediction tools: Potentialities and limitations,” *Msystems*, vol. 5, no. 4, pp. 10–1128, 2020.  
4. W. Su *et al.*, “PPD: a manually curated database for experimentally verified prokaryotic promoters,” *Journal of Molecular Biology*, vol. 433, no. 11, pp. 166860–166861, 2021.  
5. V. H. Tierrafria *et al.*, “RegulonDB 11.0: Comprehensive high-throughput datasets on transcriptional regulation in *Escherichia coli* K-12,” *Microbial Genomics*, vol. 8, no. 5, pp. 833–834, 2022.  
6. H. Dalla-Torre *et al.*, “The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics,” *bioRxiv*, 2023.  
7. Y. Lin *et al.*, “LoRA Dropout as a Sparsity Regularizer for Overfitting Control,” *arXiv preprint arXiv:2404.09610*, 2024.  
8. E. J. Hu *et al.*, “LoRA: Low-Rank Adaptation of Large Language Models,” *arXiv preprint arXiv:2106.09685*, 2021.  
9. A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” *Proceedings of the 22nd International Conference on Machine Learning*, 2005, pp. 625–632.  
10. D. S. Wilks, “On the combination of forecast probabilities for consecutive precipitation periods,” *Weather and Forecasting*, vol. 5, no. 4, pp. 640–650, 1990.  
11. J. Clauwaert, G. Menschaert, and W. Waegeman, “Explainability in transformer models for functional genomics,” *Briefings in Bioinformatics*, vol. 22, no. 5, p. bbab60, 2021.  
12. H. Huang, Y. Wang, C. Rudin, and E. P. Browne, “Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization,” *Communications Biology*, vol. 5, no. 1, pp. 719–720, 2022.  
13. A. Tareen and J. B. Kinney, “Logomaker: Beautiful sequence logos in Python,” *Bioinformatics*, vol. 36, no. 7, pp. 2272–2274, 2020.  
