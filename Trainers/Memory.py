from random import randint
import heapq


class Memory(object):
    def __init__(self, memory_size):
        self.cur_p = 0
        self.memory_instances = []
        self.memory_size = memory_size
        self.zero_cnt = 0

    def get_size(self):
        return len(self.memory_instances)

    def store(self, instance, zero=False):
        if zero:
            M = len(self.memory_instances)
            N = self.zero_cnt
            if M - N < 0.1 * N:  # too little elements
                return
        if len(self.memory_instances) < self.memory_size:
            self.memory_instances.append(instance)
        else:
            self.memory_instances[self.cur_p] = instance
        self.cur_p = (self.cur_p + 1) % self.memory_size

    def sample(self, n_instance):
        ret = []
        for i in range(n_instance):
            k = randint(0, len(self.memory_instances) - 1)
            ret.append(self.memory_instances[k])
        return ret


class Node(object):
    def __init__(self, id, instance):
        self.id = id
        self.instance = instance

    def __lt__(self, other):
        return self.id < other.id


class Memory2(object):
    def __init__(self, memory_size):
        self.cur_p = 0
        self.memory_instances = []
        self.memory_size = memory_size
        self.all_cnt = 0

    def get_size(self):
        return len(self.memory_instances)

    def store(self, instance):
        self.all_cnt += 1
        node = Node(self.all_cnt, instance)
        heapq.heappush(self.memory_instances, node)
        if len(self.memory_instances) > self.memory_size:
            heapq.heappop(self.memory_instances)

    def sample(self, n_instance):
        ret = []
        for i in range(n_instance):
            k = randint(0, len(self.memory_instances) - 1)
            ret.append(self.memory_instances[k].instance)
        return ret
