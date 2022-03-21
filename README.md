# A-Domain-adaptive-Pre-training-Approach-for-Language-Bias-Detection-in-News
The repository contains the data files and scripts corresponding to the paper "A Domain-adaptive Pre-training Approach for Language Bias Detection in News".

The models can be found anonymously on: https://drive.google.com/drive/u/4/folders/1-A1hGKeu-27X9I4ySkja5vMlVscnF8GR
- "DA-Roberta.bin": Domain-adaptive Pre-training with RoBERTa.
- "DA-T5.bin": Domain-adaptive Pre-training with T5.
- "DA-BERT.bin": Domain-adaptive Pre-training with BERT.
- "DA-BART.bin": Domain-adaptive Pre-training with BART.
- "classifier.weights.pt" and "classifier.bias.pt": Parameters for classification layer + bias used for all models prior to Domain-adaptive Pre-training to achieve maximum comparability between approaches.

# Description of files
- "BABE.xlsx": BABE corpus provided by Spinde et al. (2021): https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE
- "domain-adaptive-pretraining.ipynb": Domain-adaptive Pre-training script (More information provided in the script's header)
- "fine-tune-and-evaluate-domain-adaptive-pretraining.ipynb": Script to fine-tune Domain-adapted models on BABE corpus via 5-fold CV.

# Cite as

```
@InProceedings{Krieger2022,
  author={Krieger, David and Spinde, Timo and Ruas, Terry and Kulshrestha, Juhi and Gipp, Bela},
  booktitle={2022 ACM/IEEE Joint Conference on Digital Libraries (JCDL)}, 
  title={A Domain-adaptive Pre-training Appraoch for Language Bias Detection in News}, 
  year={2022},
  address = "Cologne,Germany"
  }
  ```
  
  More about our work can be found here: https://media-bias-research.org/


