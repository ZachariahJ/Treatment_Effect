from train import (Models, train_all, evaluate_propensity, evaluate_outcome, 
                   evaluate_ite, evaluate_denoiser)
from model import sample_C
from dataset import data_preprocessing
from config import set_seed

def main():
    set_seed()
    data = data_preprocessing()
    models = train_all(data,)
    evaluate_propensity(models.inp, models.enc, models.prop, data.test_dl)


if __name__ == "__main__":
    main()