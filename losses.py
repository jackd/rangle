import numpy as np
import tensorflow as tf


def angle_diff(angle):
    """Get the angle in the range [0, pi]."""
    angle = tf.abs(angle)
    return tf.minimum(angle, 2*np.pi - angle)


def earth_mover_loss(probs, angles, label):
    """
    Get the earth-mover loss for the specified angles.

    `probs[i]` is the inferred probability of angle `angles[i]`.
    """
    distance = angle_diff(angles - tf.expand_dims(label, axis=1))
    return tf.reduce_sum(distance*probs)


def cos_between(u, v, axis=-1):
    """
    Get the cos of the angle between unit vectors u and v.

    cos_theta = u dot v.

    Args:
        u: shape [s0, s1, s2, ..., sN, ..., sM], sN on `axis` dimension
        v: tensor, same shape as u or broadcastable to the same shape. Valid
            iff u*v is valid.

    Returns:
        cos_theta: shape [s0, s1, s2, ..., s(N-1), s(N+1), ..., sM]

    Does not check if vectors are unit vectors.
    """
    return tf.reduce_sum(u*v, axis=axis)
