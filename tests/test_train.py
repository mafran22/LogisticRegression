from model.train import train_model

def test_model_accuracy():
    acc = train_model()
    assert acc > 0.8
