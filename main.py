from model.din import DIN
from trainer.ctr_trainer import CTRTrainer

model = DIN(features=features, history_features=history_features, target_features=target_features, mlp_params={"dims":[256,128]},attention_mlp_params={"dims": [256, 128]})
