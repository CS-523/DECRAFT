# DECRAFT

A Financial Fraud Detection Framework.

Source codes implementation of papers:

- `DECRAFT`: A Decentralized Graph Learning Framework with Privacy Protection for Cryptocurrency Transaction Fraud
  Detection.

## Usage

### Decentralized Layered Model Aggregation Method

1. run `python src/layered_aggregation.py -e stop_condition` to evaluate the rate of further dividing a cluster with
   different numbers of nodes.
2. run `python src/layered_aggregation.py -e two_layer` to obtain the time consumption of two-layer model aggregation
   framework.
3. run `python src/layered_aggregation.py -e layered` to obtain the time consumption comparisons between the two-layer
   model aggregation framework and the decentralized layered model aggregation framework.

### Data Description

There are three datasets, YelpChi, Amazon and S-FFSD, utilized for model experiments in this repository. We evaluate the
proposed DECRAFT method by using three fraud detection datasets: Yelp-Fraud (Yelp), Amazon-Fraud (Amazon), and Elliptic.
The Yelp dataset is a graph dataset derived from the Yelp spam review dataset. In Yelp, 32 handcrafted features are used
as the raw node features. The Amazon dataset is a graph dataset built upon the Amazon review dataset. In Amazon, 25
handcrafted features are used as the raw node features. The Elliptic dataset is a transaction graph collected from the
Bitcoin blockchain. In Elliptic, 166 handcrafted features are associated with each node. The first 94 features represent
local information about the transaction (_e.g._, the number of inputs/outputs, transaction fee, output volume), and the
remaining 72 features are aggregated features (_e.g._, the maximum, minimum, standard deviation and correlation
coefficients of the transactions). We can use the three datasets for evaluating the performance of graph-based node
classification, fraud detection, and anomaly detection models.
 
### Data processing

1. Download the three dataset into the `data` folder：
    1. [Amazon](https://paperswithcode.com/dataset/amazon-fraud)
    2. [YelpChi](https://paperswithcode.com/dataset/yelpchi)
    3. [elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. Run `unzip -d data data/Amazon.zip` and `unzip -d data  data/YelpChi.zip` to unzip the datasets;
3. Run `python src/data_process.py`to pre-process all datasets needed in this repo.

### Training & Evalutaion

Configuration files can be found in `config/yelp_cfg.yaml`, `config/Amazon_cfg.yaml` and `config/elliptic_cfg.yaml`,
respectively.

To test implementations of `DECRAFT(center)`, `DECRAFT(without privacy protection)` and `DECRAFT`, run

```
python main.py --method "DECRAFT(center)"
python main.py --method "DECRAFT(wtp)"
python main.py --method "DECRAFT"

```


## Test Result

The performance of five models tested on three datasets are listed as follows:
|                   Methods                   |  Yelp  |        | Amazon |        | Elliptic |        |
|:-------------------------------------------:|:------:|:------:|:------:|:------:|:--------:|:------:|
|                                             |   AUC  |   F1   |   AUC  |   F1   |    AUC   |   F1   |
|                DECRAFT(center)              |  0.89  |  0.77  |  0.97  |  0.92  |   0.99   |  0.96  |
|                   DECRAFT                   | {0.88} | {0.76} | {0.97} | {0.92} |  {0.98}  | {0.95} |

## Repo Structure

The repository is organized as follows:

- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly
  use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for different models;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: organize all models;
- `requirements.txt`: package dependencies;

## Requirements

```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
```

## Citing


