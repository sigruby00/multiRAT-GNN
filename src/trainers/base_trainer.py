import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    def __init__(self, model, lr=0.01, writer=None):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.writer = writer

    @abstractmethod
    def train(self, train_loader, num_epochs=100):
        pass

    # @abstractmethod
    # def evaluate(self, test_loader):
    #     pass

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.eval()
        print(f"Model loaded from {path}")

    def load_model_transfer(self, model_new, path="model.pht"):
        self.model.load_state_dict(torch.load(path))
        # 모델의 각 레이어별 파라미터를 출력
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            print(f"Layer {idx}: {name}")
            print(f"  - Number of parameters: {param.numel()}")  # 파라미터 개수 출력
        # for name, child in self.model.named_children():
        #     for param in child.parameters():
        #         print(name, param)

        # 새 모델에 중간 레이어 가중치 복사
        with torch.no_grad():
            # layer.0.bias 복사
            model_new.layers[0].bias.copy_(self.model.layers[0].bias)

            # layer.2.weight 및 layer.2.bias 복사
            model_new.layers[2].weight.copy_(self.model.layers[2].weight)
            model_new.layers[2].bias.copy_(self.model.layers[2].bias)

        self.model = model_new
        print(model_new.layers[0].weight)

        print("load model weights for transfer learning")

        # print(f"Model loaded from {path}")
