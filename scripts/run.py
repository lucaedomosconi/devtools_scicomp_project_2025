import argparse
import pyclassify.utils as utils
import pyclassify.classifier as classifier

def shuffle(list1, list2):
    import random
    c = list(zip(list1, list2))
    random.shuffle(c)
    return zip(*c)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

filename = args.config
kwargs = utils.read_config(filename)
features, labels = utils.read_file(kwargs["dataset"])
features, labels = shuffle(features, labels)
backend = kwargs["backend"]
N_sample = len(labels)
N_split = int(N_sample * 0.2)
k = kwargs['k']
kNN_class = classifier.kNN(k, backend)

training_set = features[:N_split], labels[:N_split]
predicted_test = kNN_class(training_set, features[N_split:])
errors = sum(1 for (true_label, predicted_label) in zip(labels[N_split:], predicted_test) if true_label!=predicted_label)
print(f'Accuracy of the kNN classifier with k = {k}: {100*(1-errors/(N_sample-N_split)):.4f}%\n')

