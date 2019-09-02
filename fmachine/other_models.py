"""
Other recommendation models, for comparison.
"""

from tensorflow import keras


def keras_cf(user_map, item_map,
             item_k: int=10,
             user_k: int=10,
             reg: float=2.0,
             lr: float=0.25):

    user_input = keras.layers.Input(shape=(1,),
                                    name='user_input')
    item_input = keras.layers.Input(shape=(1,),
                                    name='item_input')

    user_embed = keras.layers.Embedding(len(user_map), user_k,
                                        embeddings_regularizer=keras.regularizers.l2(reg),
                                        name='user_embed')(user_input)
    item_embed = keras.layers.Embedding(len(user_map), item_k,
                                        embeddings_regularizer=keras.regularizers.l2(reg),
                                        name='item_embed')(item_input)

    cf = keras.layers.Dot(axes=2)([user_embed, item_embed])

    user_bias = keras.layers.Embedding(len(user_map), 1,
                                       name='user_bias')(user_input)
    item_bias = keras.layers.Embedding(len(item_map), 1,
                                       name='item_bias')(item_input)

    cf_bias = keras.layers.Add()([cf, user_bias, item_bias])
    output = keras.layers.Flatten()(cf_bias)

    model = keras.Model(inputs=[user_input, item_input],
                        outputs=output)

    model.compile(loss='mse',
                  optimizer=keras.optimizers.SGD(lr=lr,
                                                 momentum=0.9))

    return model


if __name__ == "__main__":

    from fmachine.data.movie_lens import load_move_lens_100k
    from fmachine.data.generate import IndexMap
    import numpy as np

    train = load_move_lens_100k(path='../data/ml-100k/ua.base')
    test = load_move_lens_100k(path='../data/ml-100k/ua.test')

    cf = keras_cf(train.user_id, train.item_id,
                  user_k=40,
                  item_k=40,
                  lr=0.25,
                  reg=6)

    cf.fit({'user_input': train.user_id, 'item_input': train.item_id},
           train.rating - train.rating.mean(),
           epochs=200,
           batch_size=30000,
           validation_split=0.2,
           shuffle=True)

    train_preds = cf.predict({'user_input': train.user_id, 'item_input': train.item_id})
    test_preds = cf.predict({'user_input': test.user_id, 'item_input': test.item_id})

    print(f"MSE Train: {np.mean(((train_preds.squeeze() + train.rating.mean()) - train.rating) ** 2)}")
    print(f"MSE Test: {np.mean(((test_preds.squeeze() + train.rating.mean()) - test.rating) ** 2)}")
