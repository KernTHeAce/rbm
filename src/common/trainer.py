import torch


class BaseTrainer:
    def __init__(self, optimizer_class, lr, loss, device, train_loader, test_loader, preprocessing=None):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.optimizer = None
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.preprocessing = preprocessing
        self.device = device

    def preprocess_input(self, data):
        if self.preprocessing:
            return self.preprocessing(data)
        return data.to(self.device)

    def init_optimizer(self, model):
        self.optimizer = self.optimizer_class(params=model.parameters(), lr=self.lr)

    def epoch(self, model):
        inputs, outputs = [], []
        for data in self.train_loader:
            input_ = self.preprocess_input(data)

            output = model(input_)
            loss = self.loss(output, input_)
            inputs.append(input_)
            outputs.append(output)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return model, inputs, outputs

    def test(self, model):
        inputs, outputs = [], []
        with torch.no_grad():
            for data in self.test_loader:
                input_ = self.preprocess_input(data)

                output = model(input_)
                inputs.append(input_)
                outputs.append(output)

        return inputs, outputs