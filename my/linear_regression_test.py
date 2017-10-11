# -*- coding:utf-8 -*-

"""
@author: Yan Liu
@file: linear_regression_test.py
@time: 2017/10/11 20:10
"""

import tensorflow as tf


def train():
    # X and Y data
    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # hypothesis H = WX + b
    H = X * W + b

    # 损失函数
    cost = tf.reduce_mean(tf.sqrt(H - Y))

    # 采用梯度下降更新权重
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cost)

    # 运行计算图执行训练操作
    # 在一个会话中加载运行图
    sess = tf.Session()
    # 初始化全局的变量（W、b）
    sess.run(tf.global_variables_initializer())

    # 训练、拟合
    for step in range(2001):
        cost_val, W_val, b_val, _ = \
            sess.run([cost, W, b, train], feed_dict={X: [1.1, 2.02, 2.99], Y: [1, 2.2, 3.11]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)


if __name__ == '__main__':
    train()