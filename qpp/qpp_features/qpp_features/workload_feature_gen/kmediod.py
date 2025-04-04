import numpy as np
import pylab as pl
import jellyfish
import time, os


# A version of code from https://github.com/rhasan/query-performance/blob/master/clustering/k_mediods.py
class kMediod:
    def __init__(self, queries, distance):
        self.queries = queries
        self.dist = distance

    def initial_random_centers(self, X, K):
        randidx = np.random.permutation(range(np.size(X, 0)))
        if len(X.shape) == 1:
            centers = X[randidx[0:K]]  # , :
        else:
            centers = X[randidx[0:K], :]
        return (centers, randidx[0:K])

    def find_closest_centers(self, X, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        m = np.size(X, 0)
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            min_d = np.inf
            min_j = -1
            for j in center_idxs:
                if j < 0:
                    continue

                d = distance_matrix[i, j]
                if min_d > d:
                    min_d = d
                    min_j = j
            idx[i] = min_j
            if min_j == -1:
                raise Exception()

        for j in center_idxs:
            idx[j] = j
        return idx

    def compute_centers(self, X, idx, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        moved_centers = np.zeros(K, dtype=int)
        i = 0
        for k in center_idxs:
            (x_indxs,) = np.where(idx[:] == k)
            # print "center:", k,"=",x_indxs
            # print "vals", X[x_indxs,:]

            min_cost = np.inf
            min_c = -1
            for c_indx in x_indxs:
                cost = 0.0
                for y_indx in x_indxs:
                    cost += distance_matrix[c_indx, y_indx]
                    # print "c_indx:", c_indx, "y_indx:",y_indx
                    # print "dist:", distance_matrix[c_indx,y_indx]
                # pri    nt "cost:", cost
                if min_cost > cost:
                    min_cost = cost
                    min_c = c_indx

            if np.size(x_indxs, 0) > 0:
                moved_centers[i] = min_c
                i += 1

            # print "min_cost:", min_cost
        return moved_centers[0:i]

    def k_mediods(self, X, initial_center_idxs, max_iters, distance_matrix):
        m = np.size(X, 0)
        K = np.size(initial_center_idxs, 0)
        center_idxs = initial_center_idxs
        previous_center_idxs = center_idxs
        idx = np.zeros(m, dtype=int)

        for i in range(max_iters):
            idx = self.find_closest_centers(X, center_idxs, distance_matrix)
            previous_center_idxs = center_idxs
            center_idxs = self.compute_centers(X, idx, center_idxs, distance_matrix)

            if (
                np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
                and np.size(center_idxs, 0) > 1
            ):
                if (previous_center_idxs == center_idxs).all() == True:
                    break
            elif (
                np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
                and np.size(center_idxs, 0) == 1
            ):
                if previous_center_idxs == center_idxs:
                    break

        if (
            np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
            and np.size(center_idxs, 0) > 1
        ):
            if (previous_center_idxs == center_idxs).all() == False:
                idx = self.find_closest_centers(X, center_idxs, distance_matrix)
        elif (
            np.size(previous_center_idxs, 0) == np.size(center_idxs, 0)
            and np.size(center_idxs, 0) == 1
        ):
            if previous_center_idxs == center_idxs:
                idx = self.find_closest_centers(X, center_idxs, distance_matrix)

        return (center_idxs, idx)

    def model_cost(self, X, idx, center_idxs, distance_matrix):
        K = np.size(center_idxs, 0)
        total_cost = 0.0
        for k in center_idxs:
            (k_cluster_x_indxs,) = np.where(idx[:] == k)
            # print k_cluster_x_indxs
            cost = 0.0
            for x_indx in k_cluster_x_indxs:
                cost += distance_matrix[k, x_indx]
            total_cost += cost
        return total_cost

    def initial_random_centers_cost_minimization(
        self, X, K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
    ):
        min_cost = np.inf

        for i in range(random_shuffel_max_iters):
            (initial_centers, initial_center_idxs) = self.initial_random_centers(X, K)
            (center_idxs, idx) = self.k_mediods(
                X, initial_center_idxs, kmediods_max_iters, distance_matrix
            )
            total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
            if min_cost > total_cost:
                min_cost = total_cost
                min_center_idxs = center_idxs

        return (min_center_idxs, min_cost)

    def elbow_method_choose_k_with_random_init_cost_minimization(
        self, X, max_K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
    ):
        cost_array = np.zeros(max_K, dtype=float)
        for K in range(1, max_K + 1):
            (init_center_idxs, cost) = self.initial_random_centers_cost_minimization(
                X, K, distance_matrix, random_shuffel_max_iters, kmediods_max_iters
            )
            (center_idxs, idx) = self.k_mediods(
                X, init_center_idxs, kmediods_max_iters, distance_matrix
            )
            total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
            cost_array[K - 1] = total_cost

            # print_clusters(X,idx,center_idxs)
            print("cost:", total_cost, "K:", K)

        K_vals = np.linspace(1, max_K, max_K)
        pl.plot(K_vals, cost_array)
        pl.plot(K_vals, cost_array, "rx", label="distortion")
        pl.show()

    def compute_symmetric_distance(self, X):
        m = np.size(X, 0)
        dist = np.zeros((m, m), dtype=float)
        start = time.time()
        # path = 'data/experiment1/distance_matrix_concurrent.txt'
        # max_i,max_j = -1,-1
        """if os.path.exists(path):
            with open(path,'r') as f:
                for line in f.readlines():
                    spl = line.split(',')
                    try:
                        dist[int(spl[0]),int(spl[1])] = float(spl[2])
                        if max_i < int(spl[0]):
                            max_i = int(spl[0])
                        if max_j < int(spl[1]):
                            max_j = int(spl[1])
                    except:
                        pass
        f = open(path,'a')"""
        for i in range(m):
            """if i < max_i:
                continue
            if (i% 10)==0:
                print("Query {} of {} calculated \n".format(i,m))
            for j in range(i + 1, m):
                if j < max_j:
                    continue
                #if (int(time.time()-start)%10) == 0:
                print("{},{}: Time calculating distance matrix: {}\n".format(i,j,time.time()-start))
            """
            try:
                dist[i, j] = self.dist(X[i], X[j])
                dist[j, i] = dist[i, j]
                # print "distance between", X[i], "and ", X[j], ":",dist[i,j]
            except:
                dist[i, j] = np.inf
                dist[j, i] = np.inf
        return dist

    def print_clusters(self, X, idx, center_idxs):
        for k in center_idxs:
            print("Cluster:", X[k])
            print(X[idx[:] == k])

    def test_string_clustering(self, dist=jellyfish.levenshtein_distance):
        self.dist = dist
        K = 5
        random_shuffle_max_iters = 100
        kmediods_max_iters = 100
        # X =  np.array(['ape', 'appel', 'apple', 'peach', 'puppy'])
        X = np.array(
            [
                "the",
                "be",
                "to",
                "of",
                "and",
                "a",
                "in",
                "that",
                "have",
                "I",
                "it",
                "for",
                "not",
                "on",
                "with",
                "he",
                "as",
                "you",
                "do",
                "at",
                "this",
                "but",
                "his",
                "by",
                "from",
                "they",
                "we",
                "say",
                "her",
                "she",
                "or",
                "an",
                "will",
                "my",
                "one",
                "all",
                "would",
                "there",
                "their",
                "what",
                "so",
                "up",
                "out",
                "if",
                "about",
                "who",
                "get",
                "which",
                "go",
                "me",
                "when",
                "make",
                "can",
                "like",
                "time",
                "no",
                "just",
                "him",
                "know",
                "take",
                "people",
                "into",
                "year",
                "your",
                "good",
                "some",
                "could",
                "them",
                "see",
                "other",
                "than",
                "then",
                "now",
                "look",
                "only",
                "come",
                "its",
                "over",
                "think",
                "also",
                "back",
                "after",
                "use",
                "two",
                "how",
                "our",
                "work",
                "first",
                "well",
                "way",
                "even",
                "new",
                "want",
                "because",
                "any",
                "these",
                "give",
                "day",
                "most",
                "us",
            ]
        )
        distance_matrix = self.compute_symmetric_distance(X)

        (initial_centers, initial_center_idxs) = self.initial_random_centers(X, K)
        (center_idxs, idx) = self.k_mediods(
            X, initial_center_idxs, kmediods_max_iters, distance_matrix
        )
        self.print_clusters(X, idx, center_idxs)
        total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
        print("model cost: ", total_cost)

        # elbow_method_choose_k_with_random_init_cost_minimization(X,K,distance_matrix,random_shuffle_max_iters,kmediods_max_iters)

        new_K = 10

        (min_center_idxs, min_cost) = self.initial_random_centers_cost_minimization(
            X, new_K, distance_matrix, random_shuffle_max_iters, kmediods_max_iters
        )
        print("min model cost: ", min_cost)

        (center_idxs, idx) = self.k_mediods(
            X, min_center_idxs, kmediods_max_iters, distance_matrix
        )
        self.print_clusters(X, idx, center_idxs)
        total_cost = self.model_cost(X, idx, center_idxs, distance_matrix)
        print("model cost: ", total_cost)


if __name__ == "__main__":
    k = kMediod(10, lambda x, y: x)
    k.test_string_clustering()
