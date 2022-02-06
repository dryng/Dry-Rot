class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.storage = [float('inf')] * patience
        self.done_training = False
        self.minimum_loss = float('inf')
        self.best_epoch = None
        self.curr_missed = 0
        self.epoch_count = 0
    
    def training_complete(self, loss):
        if loss < self.minimum_loss:
            self.minimum_loss = loss
            self.best_epoch = self.epoch_count
        done = True
        for step in range(self.patience):
            if loss < self.storage[step]:
                done = False
                break
        if not done:
            self.shift_storage()
            self.storage[-1] = loss
            self.done_training = False
            self.epoch_count += 1
            return False
        self.done_training = True
        return True

    def training_completeV2(self, loss):
        if loss < self.minimum_loss:
            self.minimum_loss = loss
            self.best_epoch = self.epoch_count
            self.curr_missed = 0
        else:
            if self.curr_missed >= self.patience:
                return True
            else:
                self.curr_missed += 1
        self.epoch_count += 1
        return False
        
    def shift_storage(self):
        for i in range(self.patience - 1):
            self.storage[i] = self.storage[i + 1]
        
