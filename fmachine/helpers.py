import tensorflow as tf
from typing import Callable, Dict, Any, Union
import numpy as np
from fmachine.model import FactorizationMachine


def l2_loss(y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_true - y_pred))


def train_step(mod: FactorizationMachine,
               x: Union[np.ndarray, tf.Tensor],
               y_true: Union[np.ndarray, tf.Tensor],
               lr: float,
               loss_f: Callable = l2_loss,
               loss_f_kwargs: Dict[str, Any] = None) -> tf.Tensor:
    """
    Run one training iteration.


    :param mod: Model.
    :param x: Input features.
    :param y_true: True labels.
    :param lr: Learning rate.
    :param loss_f: Fuction to use to calculate loss.
    :param loss_f_kwargs: Additional kwargs to pass on to the loss function.
    :return: Current loss (as a tensor?).
    """

    if loss_f_kwargs is None:
        loss_f_kwargs = {}

    # Calculate current loss
    with tf.GradientTape() as t:
        cur_loss = loss_f(y_true=y_true,
                          y_pred=mod(x),
                          **loss_f_kwargs)

    # Get the gradients
    db, dw, dv = t.gradient(cur_loss, [mod.b, mod.w, mod.v])

    # Assign back to model
    mod.b.assign_sub(lr * db)
    mod.w.assign_sub(lr * dw)
    mod.v.assign_sub(lr * dv)

    return cur_loss
