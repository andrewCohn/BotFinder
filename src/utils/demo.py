from recorder_util import ModelResults

seed = 123
res = ModelResults("cool_model", "your-name", seed)

# configuration #
res.record_hyperparameter("lr", 0.001)
res.record_hyperparameter("epoch", 12)

res.record_training_start()
# training #
res.record_training_stop()

res.record_testing_start()
# testing #
res.record_testing_stop()

# record eval performance
y_pred = [1, 2, 3, 3, 2, 0]
y_true = [1, 0, 3, 1, 3, 0]
res.record_performance(y_pred, y_true, None)

# write out to disk
res.write("testfile.csv")
