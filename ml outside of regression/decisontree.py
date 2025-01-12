# follows the Google for developers Machine Learning Recipes example and "Math behind Decision Tree Algorithm" Medium
# article.

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]


# just iterates over every item of the row and creates a set of the row items (set inherently means no duplicates).
def uniqrow(rows, col):
    return set(row[col] for row in rows)


# uses a dict to get the labels(names) of the row objects and assigns them to a dictionary where they key is the label
# and the value is its number of occurrence.
def classcount(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, col, val):
        self.col = col
        self.value = val

    # the example is a row, uses the col value to check a particular sample question and returns true or false depending
    # on if it is correct.
    def match(self, example):
        val = example[self.col]
        if isinstance(val, (int, float)):
            return val >= self.value
        else:
            return val == self.value

    # printing helper function
    def __repr__(self):
        condition = "=="
        if isinstance(self.value, (int, float)):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.col], condition, str(self.value))


# compares the given question to the rows and compiles how many of the rows get partitioned,
# used to check the efficiency of the questioning later on in the program.
def partition(rows, question):
    truer = []
    falser = []

    for row in rows:
        if question.match(row) is True:
            truer.append(row)
        else:
            falser.append(row)
    return truer, falser


# gini is defined as the probability that a randomly chosen element would be incorrectly labeled. 1 - sum(prob^2)
def gini(rows):
    counts = classcount(rows)
    impurity = 1
    for lbl in counts:
        proboflbl = counts[lbl] / float(len(rows))
        impurity -= proboflbl ** 2
    return impurity


# defined as the parent entropy subtracted by the weighted average entropy of the children.
def infogain(left, right, curruncert):
    p = float(len(left)) / (len(left) + len(right))
    return curruncert - p * gini(left) - (1 - p) * gini(right)


# function used to find the best question to ask at a particular point in order to maximize the partition efficiency.
def findbestsplit(rows):
    bestinfogain = 0
    bestquestion = None
    curruncert = gini(rows)
    numcols = len(rows[0]) - 1

    # iterates through every feature of the examples, finds unique values of the columns (the set), and iterates through
    # every unique value found, checks the partition data that the question has on the dataset.
    for col in range(numcols):
        values = set(row[col] for row in rows)
        for value in values:
            question = Question(col, value)

            truer, falser = partition(rows, question)

            # checks if the question had any partition effect on the dataset, if not it skips to the next value.
            if len(truer) == 0 or len(falser) == 0:
                continue

            # if the question did partition, this checks how much information we gained from that particular question.
            gain = infogain(truer, falser, curruncert)

            if gain > bestinfogain:
                bestinfogain, bestquestion = gain, question

    return bestinfogain, bestquestion


class Leaf:
    # displays the node as the name of the node and the number of times it shows up in the training data with that
    # particular characteristic.
    def __init__(self, rows):
        self.prediction = classcount(rows)


class Decisionnode:
    def __init__(self, quest, trueb, falseb):
        self.question = quest
        self.trueb = trueb
        self.falseb = falseb


def buildtree(rows):
    # finds the question that produces the highest information gain and uses that to start the branching.
    gain, question = findbestsplit(rows)

    # checks to see if there are no questions that can be asked that provide info gain and makes that a leaf.
    if gain == 0:
        return Leaf(rows)

    truer, falser = partition(rows, question)

    # builds the true branch
    trueb = buildtree(truer)
    # builds the false branch
    falseb = buildtree(falser)

    return Decisionnode(question, trueb, falseb)


# displays the tree as a series of true or false statements with the leaf nodes as results if the branching ends there.
# number of leaf nodes corresponds with the probability of a correct prediction (1 / # of nodes)
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.prediction)
        return

    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.trueb, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.falseb, spacing + "  ")


def classify(row, node):
    if isinstance(node, Leaf):
        return node.prediction
    if node.question.match(row):
        return classify(row, node.trueb)
    else:
        return classify(row, node.falseb)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

my_tree = buildtree(training_data)

print_tree(my_tree)

testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

for row in testing_data:
    print ("Actual: %s. Predicted: %s" %
           (row[-1], print_leaf(classify(row, my_tree))))
