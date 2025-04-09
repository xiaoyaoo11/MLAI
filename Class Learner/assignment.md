Write a lass learner. Main methods are:

- `__init__`: with params (model, train_dataloader, test_dataloader, optimizer, loss, schedular, work_dir, pre_train=False)
- `Train`: train model and save best model to checkpoint in work_dir
- `Test`: with params is a dataset and return the accuracy
- `Inference`: params is a image_path and return the prediction for that image

Target: Optimize sample code
