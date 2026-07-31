import time

class FTOTrainingManager:
    def __init__(self, model, optimizer, logger):
        self.model = model
        self.opt = optimizer
        self.logger = logger

    def train_epoch(self, dataloader, decision_loss_fn):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for x, y, ctx in dataloader:
            y_pred = self.model(x)
            loss = decision_loss_fn(y, y_pred, ctx)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
            
        duration = time.time() - start_time
        self.logger.log({"epoch_loss": total_loss, "time": duration})
        return total_loss / len(dataloader)