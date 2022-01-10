class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.storage = [float('inf')] * patience
        self.done_training = False
        self.best_epoch = None
    
    def training_complete(self, loss):
        done = True
        for step in range(self.patience):
            if loss < self.storage[step]:
                done = False
                break
        if not done:
            self.storage = self.storage[1:].append(loss)
            self.done_training = False
            return False
        self.best_epoch = self.storage.index.min(self.storage)
        self.done_training = True
        return True
        
