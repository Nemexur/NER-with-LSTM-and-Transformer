## Comparison of Named-Entity Recognition models based on LSTM and Transformer
In this repository you can find source code for experiments with models for NER.
* How to reproduce results:
1. Run Jupyter Notebook prepare_datasets.ipynb and get all needed data.
2. Create Docker image in root:
```bash
docker build -f docker/train.Dockerfile -t ner_experiments:dev .
```
3. Run Docker container:
```bash
docker run -it --rm -p 3000:3000 -p 6006:6006 ner_experiments:dev bash
```
4. Have Fun with HyperParameters

Also there is a convenient Makefile for Docker. Just run `make experiments` and you will skip parts 2 and 3.
