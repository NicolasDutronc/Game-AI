import numpy as np
import random


class Experience_buffer:

    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        s = 'buffer\n'
        s += 'itels:\n'
        for item in self.buffer:
            s += str(item)
            s += '\n'
        s += 'size: ' + str(len(self.buffer))
        return s
            
    
    def is_full(self):
        return len(self.buffer) == self.buffer_size
    
    def add_experience(self, experience):
        if self.is_full():
            del self.buffer[0]
        self.buffer.append(experience)

    def sample(self, size):
        return np.array(random.sample(self.buffer, size))


'''
buffer = Experience_buffer(10)
for _ in range(15):
    exp = np.random.random((1, 5, 5))
    buffer.add_experience(exp)
    print(buffer)
    print()

batch = buffer.sample(7)
print(batch)
print(batch.shape)
#'''