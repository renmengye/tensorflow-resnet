import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.9.0')
from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

layers = 152

img = load_image("data/cat.jpg")

sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
for op in graph.get_operations():
    print op.name

#init = tf.initialize_all_variables()
# sess.run(init)
print "graph restored"

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob = sess.run(prob_tensor, feed_dict=feed_dict)

print_prob(prob[0])

# for v in tf.all_variables():
#     print v.name, v.get_shape()
# print tf.all_variables()
"""
scale{1,2,3,4,5}/block{1,2,3}/{a,b,c,shortcut}/{beta,gamma,moving_mean,moving_variance}
fc/{weights,bias}

----------------------------------------------------------------------------
Look up table
----------------------------------------------------------------------------
Tensorflow-ResNet                 My code
----------------------------------------------------------------------------
scale1/weights                    conv1/w
scale1/gamma                      bn1/gamma
scale1/beta                       bn1/beta
scale1/moving_mean                bn1/ema_mean
scale1/moving_variance            bn1/ema_var
----------------------------------------------------------------------------
scale{n}/block1/shortcut/weights  stage_{n-2}/shortcut/w
scale{n}/block1/shortcut/beta     stage_{n-2}/shortcut/bn/beta
scale{n}/block1/shortcut/gamma    stage_{n-2}/shortcut/bn/gamma
scale{n}/block1/moving_mean       stage_{n-2}/shortcut/bn/ema_mean
scale{n}/block1/moving_variance   stage_{n-2}/shortcut/bn/ema_var
----------------------------------------------------------------------------
scale{n}/block{m}/{a,b,c}/weights stage_{n-1}/layer_{m-1}/unit_{k}/w
scale{n}/block{m}/{a,b,c}/beta    stage_{n-1}/layer_{m-1}/unit_{k}/bn/beta
scale{n}/block{m}/{a,b,c}/gamma   stage_{n-1}/layer_{m-1}/unit_{k}/bn/gamma
scale{n}/block1/moving_mean       stage_{n-1}/layer_{m-1}/unit_{k}/bn/ema_mean
scale{n}/block1/moving_variance   stage_{n-1}/layer_{m-1}/unit_{k}/bn/ema_var
----------------------------------------------------------------------------
fc/weights                        fc/w
fc/biases                         fc/b
----------------------------------------------------------------------------
"""
vd = {}
vd['conv1/w'] = graph.get_tensor_by_name('scale1/weights:0')
vd['bn1/gamma'] = graph.get_tensor_by_name('scale1/gamma:0')
vd['bn1/beta'] = graph.get_tensor_by_name('scale1/beta:0')
vd['bn1/ema_mean'] = graph.get_tensor_by_name('scale1/moving_mean:0')
vd['bn1/ema_var'] = graph.get_tensor_by_name('scale1/moving_variance:0')
# layers = [3, 4, 6, 3]
layers_list = [3, 8, 36, 3]

for ss in xrange(2, 6):
    vd['res_net/stage_{}/shortcut/w'.format(ss - 2)] = \
        graph.get_tensor_by_name(
        'scale{}/block1/shortcut/weights:0'.format(ss))
    vd['res_net/stage_{}/shortcut/bn/beta'.format(ss - 2)] = \
        graph.get_tensor_by_name(
        'scale{}/block1/shortcut/beta:0'.format(ss))
    vd['res_net/stage_{}/shortcut/bn/gamma'.format(ss - 2)] = \
        graph.get_tensor_by_name(
        'scale{}/block1/shortcut/gamma:0'.format(ss))
    vd['res_net/stage_{}/shortcut/bn/ema_mean'.format(ss - 2)] = \
        graph.get_tensor_by_name(
        'scale{}/block1/shortcut/moving_mean:0'.format(ss))
    vd['res_net/stage_{}/shortcut/bn/ema_var'.format(ss - 2)] = \
        graph.get_tensor_by_name(
        'scale{}/block1/shortcut/moving_variance:0'.format(ss))

# for ss in xrange(2, 6):
    for ll in xrange(layers_list[ss - 2]):
        for kk, k in enumerate(['a', 'b', 'c']):
            # print ss, ll, k
            vd['res_net/stage_{}/layer_{}/unit_{}/w'.format(
                ss - 2, ll, kk)] = \
                graph.get_tensor_by_name(
                'scale{}/block{}/{}/weights:0'.format(ss, ll + 1, k))
            vd['res_net/stage_{}/layer_{}/unit_{}/bn/beta'.format(
                ss - 2, ll, kk)] = \
                graph.get_tensor_by_name(
                'scale{}/block{}/{}/beta:0'.format(ss, ll + 1, k))
            vd['res_net/stage_{}/layer_{}/unit_{}/bn/gamma'.format(
                ss - 2, ll, kk)] = \
                graph.get_tensor_by_name(
                'scale{}/block{}/{}/gamma:0'.format(ss, ll + 1, k))
            vd['res_net/stage_{}/layer_{}/unit_{}/bn/ema_mean'.format(
                ss - 2, ll, kk)] = \
                graph.get_tensor_by_name(
                'scale{}/block{}/{}/moving_mean:0'.format(ss, ll + 1, k))
            vd['res_net/stage_{}/layer_{}/unit_{}/bn/ema_var'.format(
                ss - 2, ll, kk)] = \
                graph.get_tensor_by_name(
                'scale{}/block{}/{}/moving_variance:0'.format(ss, ll + 1, k))

vd['fc/w'] = graph.get_tensor_by_name('fc/weights:0')
vd['fc/b'] = graph.get_tensor_by_name('fc/biases:0')

# vl = []
# for k in sorted(vd.keys()):
#     vl.append(vd[k])

# rr = sess.run(vl)

# for kk, k in enumerate(sorted(vd.keys())):
#     print k, rr[kk].shape

print sess.run(graph.get_tensor_by_name("scale1/Relu:0"), feed_dict=feed_dict)
# print sess.run(graph.get_tensor_by_name("scale1/Conv2D:0"), feed_dict=feed_dict)
# print sess.run(graph.get_tensor_by_name("sub:0"), feed_dict=feed_dict)
# tf.train.Saver(vd).save(sess, 'res_{}.ckpt'.format(layers))
