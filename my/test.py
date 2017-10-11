# -*- coding:utf-8 -*-

"""
@author: Yan Liu
@file: test.py
@time: 2017/10/11 16:01
"""

import tensorflow as tf


def test1():
    """
    :return:
    """
    # 图
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(5.0)
    node3 = tf.add(node1, node2)
    print("node1:", node1, "node2", node2)
    print("node3", node3)

    sess = tf.Session()
    print("sess.run(node1, node2)", sess.run([node1, node2]))
    print("sess.run(node1, node2)", sess.run(node3))

    # 占位符和feed_dict
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add_node = a + b
    print(sess.run(add_node, feed_dict={a: 3, b: 4.5}))
    print(sess.run(add_node, feed_dict={a: [1, 3], b: [2, 4]}))

    # 张量
    # 在 TensorFlow 中，张量是计算图执行运算的基本载体，我们需要计算的数据都以张量的形式储存或声明。
    # 神经网络中一般使用float32数据类型

    # TensorFlow机器
    #  TensorFlow 机器学习模型所遵循的构建流程，即构建计算图、馈送输入张量、更新权重并返回输出值。

    # 关闭session，防止内存泄漏
    sess.close()


if __name__ == '__main__':
    test1()