class FTOWarmupScheduler:
    def __init__(self, total_epochs, warmup_ratio=0.2):
        self.total = total_epochs
        self.warmup = int(total_epochs * warmup_ratio)

    def get_loss_weights(self, epoch):
        if epoch < self.warmup:
            mse_w = 1.0
            decision_w = (epoch / self.warmup) * 0.5
        else:
            mse_w = max(0.1, 1.0 - (epoch / self.total))
            decision_w = 1.0
        return mse_w, decision_w