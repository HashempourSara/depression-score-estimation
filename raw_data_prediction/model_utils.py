import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras.backend as K
from keras.layers import Dense, Input, Conv2D, Reshape, Flatten, Dropout
from keras.initializers import glorot_normal
from keras.models import Model
from keras import optimizers

from tcn import TCN


def model_arch(input_dim, timesteps):
    krl_init = glorot_normal(seed=42)
    i = Input(shape=(timesteps, input_dim, 1), name='inp')
    h = Conv2D(filters=64, kernel_size=5, strides=(3, 3), activation='relu', 
              name='conv2_1', kernel_initializer=krl_init, data_format='channels_last')(i)
    h = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', 
              name='conv2_2', kernel_initializer=krl_init, data_format='channels_last')(h)
    s = K.int_shape(h)
    h = Reshape((s[1], s[2]*s[3]))(h)
    h = TCN(return_sequences=True, nb_filters=50, kernel_size=3, kernel_initializer=krl_init,
          nb_stacks=1, padding='causal', dilations=[1, 2, 4], activation='relu', name='tcn1')(h)
    h = Flatten(name='flt')(h)
    h = Dropout(0.5, name='drp1')(h)
    h = Dense(512, activation='relu', name='fc1')(h)
    h = Dense(1, name='fc2')(h)

    model = Model(inputs=i, outputs=h)
    adamopt = optimizers.adam(lr=0.0001)
    model.compile(optimizer=adamopt, loss='mse')
    
    return model


def train_model(train_x, train_y):
    sc = StandardScaler()
    # standardize data
    for i in range(train_x.shape[0]):
        sc.partial_fit(train_x[i])
    for i in range(train_x.shape[0]):
        train_x[i] = sc.transform(train_x[i])
    # prepare shape of data
    (_, timesteps, num_feats) = train_x.shape
    train_x = np.expand_dims(train_x, axis=-1)
    # define model
    K.clear_session()
    model = model_arch(num_feats, timesteps)
    # train model
    model.fit(train_x, train_y, batch_size=7, epochs=50,
              shuffle=True, verbose=0)
    
    return model, sc


def evaluate_10_folds(X, Y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

    train_mse, train_rmse, train_mae, train_r2 = [], [], [], []
    test_mse, test_rmse, test_mae, test_r2 = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        # select training and test data
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = Y[train_index], Y[test_index]

        model, scaler = train_model(train_x, train_y)
        
        for i in range(test_x.shape[0]):
            test_x[i] = scaler.transform(test_x[i])
        
        # prepare shape of data
        train_x = np.expand_dims(train_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)
        # model prediction
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)
        # Evaluation
        # calculate MSE
        train_mse.append(mean_squared_error(train_y, train_pred))
        test_mse.append(mean_squared_error(test_y, test_pred))
        # calculate RMSE
        train_rmse.append(np.sqrt(mean_squared_error(train_y,train_pred)))
        test_rmse.append(np.sqrt(mean_squared_error(test_y,test_pred)))
        # calculate MAE
        train_mae.append(mean_absolute_error(train_y,train_pred))
        test_mae.append(mean_absolute_error(test_y,test_pred))
        # calculate R2
        train_r2.append(r2_score(train_y,train_pred))
        test_r2.append(r2_score(test_y,test_pred))

    print('train MSE:', np.mean(train_mse), '  **  test MSE:', np.mean(test_mse))
    print('train RMSE:', np.mean(train_rmse), '  **  test RMSE:', np.mean(test_rmse))
    print('train MAE:', np.mean(train_mae), '  **  test MAE:', np.mean(train_mae))
    print('train R2:', np.mean(train_r2), '  **  test R2:', np.mean(test_r2))
