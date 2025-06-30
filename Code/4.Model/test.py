#
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../"))

import pandas as pd
from estimators.dann import H2oDANN
from module.hyper_tuning import HyperDANN

import ray

# a = ray.put("/dssg05/InternalResearch05/sheny/Mercury/2024-02-25_Remove-Platform-Bias-DANN/CleanTable/Dataset/Pre2/Valid.FragMa.Y.pt")
# print(type(a) == ray._raylet.ObjectRef)
#

# # /dssg/home/sheny/test/gsml.ttt
# model = H2oDANN()
# model.init_framer(f_feature="/dssg/home/sheny/test/gsml.cnv",
# 				  f_train="/dssg/home/sheny/test/gsml.ttt",
# 				  scale_method="minmax",
# 				  na_strategy="mean",
# 				  )
#
# init_params = {"out1": 5, "conv1": 2, "pool1": 2, "drop1": 0.1, "out2": 8, "conv2": 2,
# 			   "pool2": 2, "drop2": 0.2, "fc1": 200, "fc2": 100, "drop3": 0.2}
#
# model.train(
# 	f_train="/dssg/home/sheny/test/gsml.ttt",
# 	f_valid="/dssg/home/sheny/test/gsml.hhh",
# 	f_feature="/dssg/home/sheny/test/gsml.cnv",
#     # early_strategies=["valid|loss_class,10,min,0.1", "valid|loss_domain,10,min,0.1"],
# 	d_output="/dssg/home/sheny/test/gsml.dann",
# 	model_name="DANN",
# 	init_params=init_params,
# 	lr=0.01,
# 	weight_decay=0.001,
# 	lambda_domain=0.1,
# 	batch_size=100,
# 	epochs=500,
# )
#
# model.train(
# 	f_train="/dssg/home/sheny/test/gsml.ttt",
# 	f_valid="/dssg/home/sheny/test/gsml.hhh",
# 	f_feature="/dssg/home/sheny/test/gsml.cnv",
#     early_strategies=["valid|loss_class,10,min,0", "valid|loss_domain,10,min,0.1"],
# 	d_output="/dssg/home/sheny/test/gsml.dann",
# 	model_name="DANN",
# 	init_params=init_params,
# 	lr=0.01,
# 	weight_decay=0.001,
# 	lambda_domain=0.1,
# 	batch_size=100,
# 	epochs=500,
# )

hd = HyperDANN()

hd.run(f_train="/dssg/home/sheny/test/gsml.ttt",
	   f_valid="/dssg/home/sheny/test/gsml.hhh",
	   f_feature="/dssg/home/sheny/test/gsml.cnv",
	   f_hyper_params="/dssg/home/sheny/MyProject/gsml/config/hyper_turning/hyper_conf_dann.yaml"
	   )
