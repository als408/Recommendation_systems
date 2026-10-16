# Recommendation_systems


## Dataset train_test split

  # Download train and test splits

The train and test datasets are stored on Google Drive because they are too large to be stored directly in the GitHub repository.

### Download the datasets

```python
import pandas as pd
import gdown

ID_TRAIN = "1zjZe-u52WdE4obfOj8c1foCysYuJvfjB"
ID_TEST = "1y2AMXgEhL3NCT-0Z0nydxSJ6hI_wnrYK"

# Download train split
gdown.download(
    "https://drive.google.com/uc?id=" + ID_TRAIN,
    "df_train.csv",
    quiet=False
)

# Download test split
gdown.download(
    "https://drive.google.com/uc?id=" + ID_TEST,
    "df_test.csv",
    quiet=False
)

# Load datasets
df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")
```


### Files

After running the code, the following files will be available in your working directory:

```text
df_train.csv
df_test.csv
```



## MCRS
  @ala
  @zuza
  
## SASRec

Źródło architektury: https://github.com/kang205/SASRec

Implementacja PyTorch znajduje się w `src/recommendationsystems/sasrec.py`.
Trening i ewaluację (z taką samą definicją Precision@K i Recall@K jak dla
MCRS w notebooku) można uruchomić poleceniem:

```bash
uv run python main.py --epochs 10 -k 10
```

Model korzysta wyłącznie z chronologicznych sekwencji ze zbioru treningowego.
Podczas rekomendowania odfiltrowuje produkty widziane w treningu, a trafienia
porównuje ze zbiorem testowym dla użytkowników obecnych w obu podziałach.

Wynik dla `K=10`, 10 epok i ziarna losowego 42 (44 316 użytkowników):

```text
Precision@10: 0.0259
Recall@10:    0.0616
```
  
  @michał



## Metrics and summary 
  
