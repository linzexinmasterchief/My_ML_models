import copy

class Node:

    def __init__(self, dataset):
        self.left = None
        self.right = None
        
        # D(t) is the set of data points in the node t
        self.dataset = dataset
        # this node split on which dimension
        self.split_method = 0
        self.split_pos = 0
        # the predict label of node
        self.label = None

    def __str__(self):
        return "a"


# D = [[x1, x2, x3, ..., xn], [y1, y2, y3, ..., yn]] (D is the dataset imported)
# x is data value, y is data label
D = [[1, 2, 3, 4, 5], [3, 3, 4, 4, 5]]

def get_unique_label_from_dataset(dataset):
    label = dataset[1]
    unique = []
    for l in label:
        if (l in unique):
            continue
        else:
            unique.append(l)
    return unique

# the subset in dataset D which satisfies label j
def D_j(D, j):
    # D[1] is label set Y, range(len(D[1])) is [0, 1, 2, 3, 4, ...], in another words, the index of D[1]
    # therefore the returned i is an array of index that matchs j
    return [x_i for x_i, x_j in zip(range(len(D[1])), D[1]) if x_j == j]

# N(t) is the number of data points in the node t
def N(t):
    return len(t.dataset)

# N(t, j) is the number of data points in the node t with class label j
# D(t) is t.dataset
def N(t, j):
    return len(D_j(t.dataset, j))

# prior probability distribution of U that represents the probability of picking the class label j
# here use pi(j) = Nj / N
pi = {}
for label in get_unique_label_from_dataset(D):
    pi.update({label : len(D_j(D, label)) / len(D[1])})
    # [len(D_j(D, j)) / len(D[1]) for j in D[1]]
# probability of picking up a data point that is in t with the class label j p(j, t)
def p_j_t(j, t):
    return pi.get(j) * N(t, j) / len(D_j(t.dataset, j))

# probability of picking a data point in the node t
def p_t(t):
    return sum([p_j_t(j, t) for j in t.dataset[1]])

# conditional probability of picking a data point with the class label j when the node t is given p(j | t)
def p_j_if_t(j, t):
    return p_j_t(j, t) / p_t(t)

# impurity function
def fi(p):
    # use Gini impurity model
    return 1/2 * sum([i / (1.00001 - i) for i in p])

# impurity measure of a node i(t)
def fi_t(t):
    return fi([p_j_if_t(j, t) for j in t.dataset[1]])

# impurity of a node t I(t)
def fI_t(t):
    return fi_t(t) * p_t(t)

# get an array of nodes that are under tree root t
def get_nodes_under_root(t):
    T = []
    T.append(t)
    if (t.left): T.extend(get_nodes_under_root(t.left))
    if (t.right): T.extend(get_nodes_under_root(t.right))
    return T

# impurity of a tree T I(T), T is the tree root
def fI_T(T):
    return sum([fI_t(t) for t in get_nodes_under_root(T)])


def total_impurity_change(original_set, left_set, right_set):
    # impurity of original node
    current_node = Node(original_set)
    Impurity_current_node = fI_t(current_node)
    # impurity of left node
    left = Node(left_set)
    Impurity_left_node = fI_t(left)
    # impurity of right node
    right = Node(right_set)
    Impurity_right_node = fI_t(right)
    # delta I(s, t), the total impurity change due to split s
    return (Impurity_current_node - Impurity_left_node - Impurity_right_node)
    

# get the least impurity split
# return node t with splited left and right node, the complement piece can be gain easily by array diff operation
# min_data_amount is the minimum a mount of data allowed in one piece of split
def min_impurity_split(t, min_data_amount):
    dataset = copy.deepcopy(t.dataset)
    max_impurity_change = 0
    # init best_split set to be all data on one piece
    best_split = [[[], []], dataset]
    split_method = 0
    split_pos = 0

    for dimension in range(len(t.dataset)):
        # data is the split piece (left piece)
        data = [[] for i in range(len(t.dataset))]
        dataset = copy.deepcopy(t.dataset)
        print("[dimension", dimension, "]", t.dataset, "|", data, "|", dataset)
        for data_index in range(len(t.dataset[dimension])):
            # get index of min data x
            i = dataset[dimension].index(min(dataset[dimension]))
            for dimension_2 in range(len(dataset)):
                # extract data point and append to left piece
                data[dimension_2].append(dataset[dimension_2][i])
                # remove data point from the rest
                dataset[dimension_2].pop(i)
                print("        [dataset cut]", "data: ", data, "dataset: ", dataset)
            print("    [impurity parameter]", t.dataset, "|", data, "|", dataset)

            # there must be at lest [min_data_amount] of data point in data / dataset
            if (len(data[dimension]) < min_data_amount or len(dataset[dimension]) < min_data_amount -1):
                continue


            current_impurity_change = total_impurity_change(t.dataset, data, dataset)
            
            print("    [impurity]", current_impurity_change)
            if (-current_impurity_change > max_impurity_change):
                
                max_impurity_change = -current_impurity_change
                best_split = copy.deepcopy([data, dataset])
                split_method = dimension
                # split at the data point position (should have a better approach)
                split_pos = dataset[dimension][i]



    left = Node(best_split[0])
    right = Node(best_split[1])
    t.left = left
    t.right = right
    t.split_method = split_method
    t.split_pos = split_pos
    return t

# node t, assume label i
def misclassification_cost(t):
    max_cost = 0
    unique_label = get_unique_label_from_dataset(t.dataset)
    
    for i in unique_label:
        cost = p_j_if_t(i, t)
        if cost > max_cost:
            max_cost = cost
    return max_cost

def class_label_of_node(t):
    dataset = t.dataset
    unique_label = get_unique_label_from_dataset(dataset)
    # majority vote is a label in the label set of node t with max amount
    majority_vote = unique_label[0]
    max_label_count = 0
    for i in unique_label:
        if (dataset[1].count(i) > max_label_count):
            max_label_count = dataset[1].count(i)
            majority_vote = i
    return majority_vote

def node_test():
    # root node contain whole dataset
    root = Node(D)
    a = Node(D)
    b = Node(D)
    c = Node(D)
    d = Node(D)
    e = Node(D)
    f = Node(D)

    root.left = a
    root.right = b

    a.left = c
    a.right = d

    b.left = e
    b.right = f

    print(N(root, 3))
    print(pi)
    print(get_nodes_under_root(root))
    print(class_label_of_node(root))

def data_test():
    u = get_unique_label_from_dataset(D)
    print(u)

def train(t, node_count):
    node_count += 1
    # if terminal node (only contain one data point), stop spliting
    if (len(t.dataset[0]) <= 2):
        return t
    if (misclassification_cost(t) < 0.1):
        return t
    # split
    print("============= new node =============")
    print("node", node_count, ": ", t.dataset)
    t = min_impurity_split(t, 2)
    if (t.left):
        train(t.left, node_count)
    if (t.right):
        train(t.right, node_count)
    return t


def print_tree(t):
    print(t.dataset, end="")
    if (t.right):
        print(" ------> ", end="")
        print_tree(t.right)
    if (t.left):
        print("\n  |")
        print_tree(t.left)
    

root = Node(D)
train(root, 0)
print("=====================")
print_tree(root)