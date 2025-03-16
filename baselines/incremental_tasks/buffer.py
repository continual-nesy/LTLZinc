import heapq as hq
import numpy as np

class RingBuffer:
    """
    Class-balanced ring buffer class. It stores the last buffer_size // num_observed_classes observed samples for each class, and returns
    a random minibatch of samples from every class.
    Whenever a new class is observed, balancing is guaranteed by discarding the oldest samples from classes exceeding the new capacity.
    """
    def __init__(self, buffer_size, rng):
        self.rng = rng
        self.bins = {}
        self.max_capacity = buffer_size

    def __len__(self):
        if len(self.bins.keys()) == 0:
            return 0
        else:
            return sum([len(b["data"]) for b in self.bins.values()])

    def reset(self):
        self.bins = {}

    def observe(self, batch):
        for i in range(len(batch[0])):
            x = batch[0][i], batch[1][i],batch[2][i]
            var_label = x[1].item()
            # Observed a new label: delete old samples from other bins if necessary, and then initialize new bin.
            if var_label not in self.bins:
                new_capacity = self.max_capacity // (len(self.bins.keys()) + 1)

                for v in self.bins.values():
                    if len(v["data"]) > new_capacity:
                        if len(v["data"]) < self.max_capacity // len(self.bins.keys()):
                            # If the buffer is not full yet, keep only the last new_capacity samples.
                            v["data"] = v["data"][max(0, len(v["data"]) - new_capacity):]
                            v["ptr"] = len(v["data"])
                        else:
                            # If the buffer is full, remove old_capacity - new capacity samples from current position.
                            if v["ptr"] >= new_capacity: # All recent samples lie before the pointer.
                                v["data"] = v["data"][v["ptr"] - new_capacity: v["ptr"]]
                                v["ptr"] -= new_capacity
                            else: # Some old sample crosses list index 0.
                                v["data"] = v["data"][:v["ptr"]] + v["data"][v["ptr"] + len(v["data"]) - new_capacity:]

                    # Re-roll the pointer to 0, in case it fell exactly at new_capacity.
                    v["ptr"] = v["ptr"] % (new_capacity)

                self.bins[var_label] = {"data": [], "ptr": 0}

            # Add observed value.
            if len(self.bins[var_label]["data"]) < self.max_capacity // len(self.bins.keys()):
                self.bins[var_label]["data"].append(x)
            else:
                ptr = self.bins[var_label]["ptr"]
                self.bins[var_label]["data"][ptr] = x
            self.bins[var_label]["ptr"] = (self.bins[var_label]["ptr"] + 1) % (self.max_capacity // len(self.bins))

    def get_random_batch(self, batch_size):
        bin_probs = np.array([len(self.bins[k]["data"]) for k in sorted(self.bins.keys())], dtype=float)
        if bin_probs.sum() > 0:
            out = []
            bin_probs /= bin_probs.sum()
            target_bins = self.rng.choice(sorted(self.bins.keys()), batch_size, replace=True, p=bin_probs)

            for b in target_bins:
                candidates = self.bins[b]["data"]
                out.append(candidates[self.rng.choice(range(len(candidates)))])

            return list(zip(*out))
        else:
            return [[], [], []]

    def get_class_batches(self, batch_size):
        return {k:
                    list(zip(*[list(l) for l in self.rng.choice(v["data"], batch_size, replace=(len(v["data"]) < batch_size))]))
                for k, v in self.bins.items() if len(v["data"]) > 0}

class ReservoirBuffer(RingBuffer):
    """
        Class-balanced reservoir buffer class. It stores samples by means of reservoir sampling (https://en.wikipedia.org/wiki/Reservoir_sampling),
        and returns a random minibatch of samples from every class.
        """
    def observe(self, batch):
        prio = self.rng.random(len(batch[0]))
        bin_keys = sorted(self.bins.keys())

        for i in range(len(batch[0])):
            x = batch[0][i], batch[1][i], batch[2][i]
            var_label = x[1].item()

            # Observed a new label: initialize new bin.
            if var_label not in self.bins:
                self.bins[var_label] = {"data": []}

            # Add observed value.
            if len(self) < self.max_capacity:
                hq.heappush(self.bins[var_label]["data"], (float(prio[i]), x))
            else:
                remove_bin = bin_keys[np.argmax([len(self.bins[x]["data"]) for x in bin_keys])]
                if prio[i] > self.bins[remove_bin]["data"][0][0]:
                    hq.heappop(self.bins[remove_bin]["data"])
                    hq.heappush(self.bins[var_label]["data"], (float(prio[i]), x))

    def get_random_batch(self, batch_size):
        bin_probs = np.array([len(self.bins[k]["data"]) for k in sorted(self.bins.keys())], dtype=float)
        if bin_probs.sum() > 0:
            out = []
            bin_probs /= bin_probs.sum()
            target_bins = self.rng.choice(sorted(self.bins.keys()), batch_size, replace=True, p=bin_probs)

            for b in target_bins:
                candidates = [l[1] for l in self.bins[b]["data"]]
                out.append(candidates[self.rng.choice(range(len(candidates)))])

            return list(zip(*out))

        else:
            return [[], [], []]

    def get_class_batches(self, batch_size):
        return {k:
                    list(zip(*[list(l) for l in self.rng.choice([k[1] for k in v["data"]], batch_size, replace=(len(v["data"]) < batch_size))]))
                for k, v in self.bins.items() if len(v["data"]) > 0}
