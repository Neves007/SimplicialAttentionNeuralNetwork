
# 训练过程的输出
def log_train_begin(train_params):
    train_info = '_'.join((str(value) for value in train_params))
    print("Begin to train")
    print("train model:",train_info)

def log_training():
    print("training model...")

# 真实网络

def log_realnet_begin(train_params):
    train_info = '_'.join((str(value) for value in train_params))
    print("realnet model:",train_info)

def log_exp_end():
    print("\tEnd.")
    print()

def log_analyze_trained_model(train_param):
    train_info = ' '.join((str(value) for value in train_param))
    print(train_info)

