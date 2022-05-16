class DefaultConfigs(object):
    # 1.string parameters
    train_data = "./dataset/DeepLesion/train/"
    test_data = "./dataset/DeepLesion/test/"
    val_data = "./dataset/DeepLesion/val/"
    dataset = "./dataset/DeepLesion/"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"
    augmen_type = "medium"

    # 2.numeric parameters
    epochs = 4
    batch_size = 2
    img_height = 512
    img_weight = 512
    num_classes = 2
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
