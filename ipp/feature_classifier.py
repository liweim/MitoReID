from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns


def build_model(size, n_classes, hidden_cell, learning_rate):
    model = Sequential()
    model.add(LSTM(units=hidden_cell, return_sequences=True, input_shape=size))
    model.add(Dropout(0.2))
    model.add(LSTM(units=hidden_cell, return_sequences=True))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=hidden_cell, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(units=hidden_cell))
    # model.add(Dropout(0.2))
    model.add(Dense(units=n_classes))
    model.add(Activation("softmax"))
    optimizer = optimizers.adam_v2.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def train_lstm(data_path, model_path, plot_path, result_path, lr=1e-4, resume=False):
    x_train, x_test, y_train, y_test = np.load(data_path, allow_pickle=True)
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_train = x_train.reshape([len(x_train), -1, 16])
    x_train = np.asarray([x.T for x in x_train])
    x_test = x_test.reshape([len(x_test), -1, 16])
    x_test = np.asarray([x.T for x in x_test])
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(np.asarray(y_train).reshape(-1, 1)).toarray()
    y_test = encoder.fit_transform(np.asarray(y_test).reshape(-1, 1)).toarray()

    size = x_train.shape[1:]
    n_classes = y_train.shape[1]

    if resume:
        model = load_model(model_path)
        print(f'resume from {model_path}')
    else:
        model = build_model(size, n_classes, hidden_cell=32, learning_rate=lr)
        # early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=50, verbose=0, mode='max',
        #                                restore_best_weights=False)
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',
                                     period=1)
        history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=2,
                            callbacks=[checkpoint], shuffle=False)

        # train_acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # df = pd.DataFrame(train_acc, columns=["train_acc"])
        # df["val_acc"] = val_acc
        # df.to_csv(plot_path, index=None)

        # 绘制损失趋势线
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        # plt.savefig(plot_path.replace('.csv', '.png'))
        plt.legend()
        plt.show()

        model = load_model(model_path)
        result = model.predict(x_train, verbose=1)
        ys = np.argmax(y_train, 1)
        ys_pred = np.argmax(result, 1)
        evaluate(ys, ys_pred, "train")

    result = model.predict(x_test)
    ys = np.argmax(y_test, 1)
    ys_pred = np.argmax(result, 1)
    evaluate(ys, ys_pred, "val")
    pd.DataFrame(y_test).to_csv(result_path, index=None)
    pd.DataFrame(result).to_csv(result_path.replace('gt', 'pred'), index=None)


def evaluate(y_test, y_pred, type):
    y_pred = np.array(y_pred).astype(int)
    y_test = np.array(y_test).astype(int)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    precision = metrics.precision_score(y_test, y_pred, average="macro")
    F1 = 2 * recall * precision / (recall + precision)
    print(f"{type} acc\trecall\tprec\tf1")
    print(f"{accuracy}\t{recall}\t{precision}\t{F1}")


def feature_selection(xs, ys, save_path):
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute

    extracted_features = extract_features(xs, column_id="id", column_sort="time")
    print('impute')
    extracted_features = impute(extracted_features)
    print('select features')
    features_filtered = select_features(extracted_features, ys).values
    np.save(save_path, features_filtered)
    print('done')
    return features_filtered


def prepare_data(path, save_path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna()

    target_df = pd.read_excel("../utils/annotation.xlsx", sheet_name="FDA-list")[["index", 'moa_id', 'moa_choose']]
    target_df = target_df[target_df['moa_choose'] == 1]
    target_df['index'] = target_df['index'].astype(str)

    labels = []
    for y in df["label"].values:
        label = y.split("-")[0].split("^")[0]
        if label == 'control':
            label = 'dmso'
        labels.append(label)
    df["label"] = labels
    merge_df = pd.merge(df, target_df, left_on="label", right_on="index")
    merge_df = merge_df.dropna()
    ys = merge_df['moa_id'].values.astype(str)
    labels = list(set(ys))
    labels.sort()
    print(labels)
    ys = np.asarray([labels.index(y) for y in ys])
    xs = merge_df.values[:, :-6].astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.8)
    np.save(save_path, [x_train, x_test, y_train, y_test])
    return x_train, x_test, y_train, y_test


def train_ml(method, data_path, result_path):
    print(method)
    x_train, x_test, y_train, y_test = np.load(data_path, allow_pickle=True)
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    if method == 'rf':
        model = RandomForestClassifier()
    elif method == 'svc':
        model = LinearSVC()
    elif method == 'mlp':
        model = MLPClassifier()
    else:
        return

    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    evaluate(pred, y_train, "train")
    pred = model.predict(x_test)
    evaluate(pred, y_test, "test")

    # stratifiedkf = StratifiedKFold(n_splits=5)
    # score = cross_val_score(model, xs, ys, cv=stratifiedkf)
    # print("Cross Validation Scores are {}".format(score))
    # print("Average Cross Validation score :{}".format(score.mean()))

    encoder = OneHotEncoder(sparse_output=False)
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))
    pd.DataFrame(y_test).to_csv(result_path, index=None)
    result = model.predict_proba(x_test)
    pd.DataFrame(result).to_csv(result_path.replace('gt', 'pred'), index=None)


def plot_training(plot_path):
    df = pd.read_csv(plot_path)
    df["iter"] = range(len(df))
    sns.relplot(x="iter", y="train_acc", kind="line", data=df)
    plt.show()


if __name__ == "__main__":
    data_path = 'data/l123_center_zscore.npy'
    prepare_data("../paper/data/l123_center_zscore_stack.csv", save_path=data_path)

    # train_ml('svc', data_path, 'result/svc_roc_gt.csv')
    # train_ml('rf', data_path, 'result/rf_roc_gt.csv')
    # train_ml('mlp', data_path, 'result/mlp_roc_gt.csv')
    # train_lstm(data_path, 'result/lstm.h5', "plot/training_plot.csv", 'result/lstm_roc_gt.csv', lr=1e-3, resume=False)
