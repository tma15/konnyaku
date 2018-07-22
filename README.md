# Konnyaku
A PyTorch implementation of neural encoder-decoders.

## Requirements
- Python 3.6 (confirmed with Python 3.6.6. Pytho 3.5 is also maybe ok.)
- See `requirements.txt` for other libraries

## Install
```sh
pip install -r requirements.txt
python setup.py install
```

## Training
```sh
konnyaku_train -c conf.ini -s <src_file> -t <trg_file> -e <epoch> -b <batch_size> -g <device_id>
```

## Evaluation
```sh
konnyaku_eval -c conf.ini -m <model_file> -s <src_file> -t <trg_file>
```
