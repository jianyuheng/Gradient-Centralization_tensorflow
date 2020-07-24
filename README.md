# Gradient-Centralization_tensorflow
reproducing by tensorflow1.14

```
grads_and_vars = optimizer.compute_gradients(loss)
grads_GC_and_vars = []
#GC operation for Conv layers and FC layers
for (grad, param) in grads_and_vars:
    if grad is not None and param is not None:
        if len(list(grad.shape)) > 1:
            grad -=  tf.reduce_mean(grad, axis=(tuple(range(1,len(list(grad.shape))))), keep_dims=True)
    grads_GC_and_vars.append((grad, param))
#grad_summy_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads])
train_op = optimizer.apply_gradients(grads_GC_and_vars)
```
