import torch


class BaseTrainer:
    def __init__(
        self,
        optimizer_class,
        lr,
        loss,
        device,
        train_loader,
        test_loader,
        preprocessing=lambda x: x,
        postprocessing=lambda x: x,
    ):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.device = device

    def init_optimizer(self, model):
        self.optimizer = self.optimizer_class(params=model.parameters(), lr=self.lr)

    def get_data(self, batch):
        if len(batch) == 2:
            input_ = self.preprocessing(batch[0]).to(self.device)
            if isinstance(self.loss, torch.nn.CrossEntropyLoss):
                target = batch[1].to(self.device)  # for mnist classification
                return input_, target
            else:
                return input_, input_  # for mnist ae
        input_ = self.preprocessing(batch).to(self.device)
        return input_, input_

    def epoch(self, model):
        loss_sum = 0
        targets, outputs = [], []
        for batch in self.train_loader:
            input_, target = self.get_data(batch)
            output = model(input_)
            loss = self.loss(output, target)
            targets.append(target)
            outputs.append(self.postprocessing(output))
            loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return model, targets, outputs, loss_sum / len(self.train_loader)

    def test(self, model):
        loss_sum = 0
        targets, outputs = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                input_, target = self.get_data(batch)

                output = model(input_)
                targets.append(target)
                loss = self.loss(output, target)
                loss_sum += loss.item()
                outputs.append(self.postprocessing(output))

        return targets, outputs, loss_sum / len(self.test_loader)
