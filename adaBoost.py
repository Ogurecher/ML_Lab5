from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from plot import plot_results, get_z
from prepare_data import prepare_data
from train_test_split import train_test_split
import parameters

dataset_names = ['chips', 'geyser']
dataset_name = dataset_names[0]

filename = 'data/{}.csv'.format(dataset_name)

dataset, features, labels = prepare_data(filename, normalization=False)

train_test_ratio = 0.8

train_set, train_features, train_labels, test_set, test_features, test_labels = train_test_split(dataset, train_test_ratio)

results = []

for n_estimators in parameters.n_estimators:
    for learning_rate in parameters.learning_rate:
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

        model.fit(train_features, train_labels)

        predicted_labels = model.predict(test_features)

        f_score = f1_score(test_labels, predicted_labels, average='binary', pos_label='P')

        results.append({'f_score': f_score, 'n_estimators': n_estimators, 'learning_rate': learning_rate})

results = sorted(results, key=lambda k: (k['f_score'], -k['n_estimators']), reverse=True)
print(results)

best_n_estimators = results[0]['n_estimators']
best_learning_rate = results[0]['learning_rate']

model = AdaBoostClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)

model.fit(train_features, train_labels)

staged_predictions = model.staged_predict(features)

iteration = 0
staged_predictions_list = []

z = get_z(x=features[:, 0], y=features[:, 1])

staged_predict = model.staged_predict(features)
staged_predict_z = model.staged_predict(z)

for predicted_labels, zz in zip(staged_predict, staged_predict_z):
    staged_predictions_list.append(predicted_labels)

    plot_results(x=features[:, 0], y=features[:, 1], zz=zz, actual_labels=labels, predicted_labels=predicted_labels,
                 filename_out='./results/{}/iteration_{}.png'.format(dataset_name, iteration))

    iteration += 1

plt.plot([f1_score(labels, staged_predicted_labels, average='binary', pos_label='P') for staged_predicted_labels in staged_predictions_list])
plt.title('n_estimators: {}, learning_rate: {}'.format(best_n_estimators, best_learning_rate))

plt.savefig('./results/{}/metric.png'.format(dataset_name))
plt.close()