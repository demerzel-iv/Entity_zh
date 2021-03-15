class counter:
    def __init__(self, n_types : int):
        self.n = n_types
        self.clear()

    def clear(self):
        self.cnt = [[0,0],[0,0]]

    def count(self,out, ans):
        for i in range(self.n):
            self.cnt[ans[i] > 0.99][out[i] > 0.2] += 1
    
    def output(self):
        print('================================')
        print('accuracy = ', (self.cnt[0][0] + self.cnt[1][1])*1.0 / (sum(self.cnt[0]) + sum(self.cnt[1])) )
        print('precision = ', self.cnt[1][1]*1.0 / (self.cnt[0][1] + self.cnt[1][1]) )
        print('recall = ', self.cnt[1][1]*1.0 / sum(self.cnt[1]) )
        print('================================')
