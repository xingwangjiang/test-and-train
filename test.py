import tensorflow as tf#调用tf环境
import mmd_rbf_tf as mmd#导入这个文件后面叫做mmd
import numpy as np#NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库。
import scipy.io as scio#scipy库是构建在numpy的基础之上的，它提供了许多的操作numpy的数组的函数。scipy.io包提供了多种功能来解决不同格式的文件的输入和输出。

data = scio.loadmat("./data_cwru.mat")#mat数据包 #time:10000*1200./traindata_frequency.mat  #mat数据格式是Matlab的数据存储的标准格式。在Python中可以使用scipy.io中的函数loadmat()读取mat文件。
# sdata1 = data['fftdata1']
# sdata2 = data['fftdata2']
sdata = data['fftdata3']#数据3  #快速傅里叶变换的数据3  #s为源域data数据
# sdata = np.concatenate([sdata1,sdata2,sdata3])  #基于numpy库concatenate是一个非常好用的数组操作函数。把sdata1,2,3组合起来

label = data['label']#标签  #标签数据
# slabel = np.concatenate([label,label,label])
tdata = data['fftdata0']#数据1  #快速傅里叶变换的数据0  #t为目标域data数据
slabel = label  #s源域的标签
a1 = np.random.randint(1000, size=1000)#生成1000个1000以内的随机数 #random随机 1000范围 size是输出随机数组的尺寸 #这个函数作用是，返回一个随机整型数，其范围为[1000, size)。如果没有写参数size的值，则返回[size=1000,1000)的值。
# a2 = np.random.randint(1000, 10000, size=adsiza)
a2 = np.random.randint(9000, size=1000)#1000范围  size个数
a3 = np.concatenate([a1, a2+1000])#a1和a2+1000两个数组拼起来（umpy.concatenate((a1,a2,…), axis=0)函数，能够一次完成多个数组的拼接。）
tdata_ = tdata[a3, :]#14行定义过，取tdata的第a3行：是行或列？
tlable = label[a3,:]#12行定义过，赋给tlable
#以后用到的参数，23-29行
ep = 300#迭代次数
batchsize = 200#一组去多少个  #batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
inputdim = 600#输入的维度 #通俗来说，input_length就是输入数据的长度，Input_dim就是数据的维度。
l1 = 400#l1什么意思？
l2 = 100
l3 = 10
n_class = 10#分类数量#25-29参数  10种故障类型

xstr = tf.placeholder(tf.float32, [None, inputdim])#占位（float32是一数据类型，【】占位的矩阵形状none行不定义，inputdim之前定义25行
xs = tf.placeholder(tf.float32, [None, inputdim])  #palceholder 的用途： 在神经网络每次进行迭代的时候，传进来的数据都是不一样的，在使用数据之前，我们必须把数据构造好，并且把数据切成一块一块的，batchsize 每次传进来的数据都是固定的，我们可以先把这样的空间大小固定好，然后把batchsize 样的数据 传进来。
xt = tf.placeholder(tf.float32, [None, inputdim])  #tf.placeholder(dtype,shape=None,name=None)dtype：数据类型。常用的是tf.float32,tf.float64等数值类型/shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）name：名称
ys = tf.placeholder(tf.float32, [None, n_class])#29行定义
keep = tf.placeholder(tf.float32)#只定义了数据类型？
lam = tf.placeholder(tf.float32)
learn = tf.placeholder(tf.float32, [])#32-37定义了一个框子，后面放苹果或梨，都是空的

def my_init(size):#def，return定义函数
    return tf.random_uniform(size, -0.1, 0.1)#random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)shape：张量形状 minval：随机值范围下限，默认0 maxval:随机值范围上限，如果 dtype 是浮点，则默认为1 dtype：输出的类型：float16、float32、float64、int32、orint64 seed：一个 Python 整数.用于为分布创建一个随机种子 name：操作的名称(可选)
                                             #生成的值在该 [minval, maxval) 范围内遵循均匀分布。下限 minval 包含在范围内，而上限 maxval被排除在外。对于浮点数，默认范围是 [0, 1)。对于整数，至少 maxval 必须明确地指定。

W1 = tf.Variable(my_init([inputdim, l1]),dtype=tf.float32)#variable定义变量，39行，dtype 返回数据元素的数据类型（int、float等） #https://blog.csdn.net/kakiebu/article/details/113341094?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164897203116780255271983%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164897203116780255271983&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-113341094.142^v5^pc_search_result_control_group,157^v4^control&utm_term=tf.Variable&spm=1018.2226.3001.4187
b1 = tf.Variable(tf.zeros([l1]),dtype=tf.float32)#zeros矩阵里面都为0，定义一个l1400的0矩阵#tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
W2 = tf.Variable(my_init([l1 , l2]),dtype=tf.float32)
b2 = tf.Variable(tf.zeros([l2]),dtype=tf.float32)
W3 = tf.Variable(my_init([l2 , l3]),dtype=tf.float32)
b3 = tf.Variable(tf.zeros([l3]),dtype=tf.float32)
# W4 = tf.Variable(my_init([l3 , n_class]),dtype=tf.float32)
# b4 = tf.Variable(tf.zeros([n_class]),dtype=tf.float32)
variables = [ W1, W2, W3, b1, b2, b3]#6个参数放到大矩阵里，为什么这么排？#43-51定义了个初试量，跑起来，让后面数据做对比/更新

def coral(x,y):#def到return都是定义的函数的内容  核函数
    d = tf.to_float(tf.shape(x)[1])
    tfxm = tf.reduce_mean(x, 1, keep_dims=True) - x#求均值
    tfxc = tf.matmul(tf.transpose(tfxm), tfxm)#对于两矩阵相乘，tf.multiply()元素相乘
    #tf.transpose数组转置
    tfxmt = tf.reduce_mean(y, 1, keep_dims=True) - y
    tfxct = tf.matmul(tf.transpose(tfxmt), tfxmt)
    tfloss = tf.reduce_sum((tfxc - tfxct) ** 2) / (4 * d * d)
    return tfloss

def mean_kernal(x: object, y: object, kernel_mul: object = 2.0, kernel_num: object = 5, fix_sigma: object = 1) -> object:#平均核
    fx = tf.reduce_mean(x,0)
    fy = tf.reduce_mean(y,0)
    bandwidth = fix_sigma
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [(-2*tf.exp(-tf.reduce_sum((fx-fy)**2) / (2*bandwidth_temp**2))) for bandwidth_temp in bandwidth_list]
    return tf.reduce_sum(kernel_val)

def ANN(X, k):#定义人工神经网络
    X1 = tf.nn.relu(tf.matmul(X, W1) + b1)#激活函数
    X1d = tf.nn.dropout(X1, k)
    X2 = tf.nn.relu(tf.matmul(X1d, W2) + b2)
    X2d = tf.nn.dropout(X2, k)
    X3 = tf.matmul(X2d, W3) + b3

    # X4 = tf.matmul(X3_d, W4) + b4
    return X1,X2,X3

_,_,pred = ANN(xstr,keep)#xste-X  keep-K
loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))#loss损失函数，reducemean求均值softmax 分类损失cross entropy交叉熵

p1,p2,spred = ANN(xs,1)#K=1
f1,f2,fpred = ANN(xt,1)#K=1
loss_d1 = mmd.mmd_rbf_tf(p1,f1)#最大均值差异#调用的mmd文件
loss_d2 = mmd.mmd_rbf_tf(p2,f2)

# loss_d1 = coral(p1,f1) + mean_kernal(p1,f1)
# loss_d2 = coral(p2,f2) + mean_kernal(p2,f2)

loss = loss_c + lam * ( loss_d1 + loss_d2)#lam之前定义过

solver = tf.train.AdamOptimizer(learn).minimize(loss, var_list=variables)#Optimizer优化器

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))#equal求两个矩阵中相等的equal(x, y, name=None)equa判断，x, y 是不是相等
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#求平均值


sess = tf.Session()#创建一个新的TensorFlow？
sess.run(tf.global_variables_initializer())#在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer
# saver = tf.train.Saver()
# saver.restore(sess, "./Tensflowsave/Net.ckpt")
# acc_test = sess.run(p2, feed_dict={xs: tdata,keep: 1})
# acc_train = sess.run(p2, feed_dict={xs: sdata,keep: 1})

# s1 = sess.run(W1)
# s2 = sess.run(W2)
# s3 = sess.run(W3)
#
# d1 = sess.run(b1)
# d2 = sess.run(b2)
# d3 = sess.run(b3)


for e in range(ep):#循环，ep循环次数 #for i in range ()作用：range()是一个函数， for i in range () 就是给i赋值：比如 for i in range （1，3）：
    a = np.random.randint(10000, size=batchsize)
    b1 = np.random.randint(500,size=30)
    b2 = np.random.randint(500,size=30)
    b = np.concatenate([b1,b2+1000, b2 + 2000,b2+4000,b2+5000,b2+7000,b2+8000])#concatenate19行
    # b = np.random.randint(8000, size=batchsize)
    data_s = sdata[b, :]
    data_t = tdata[b, :]

    data_tr = sdata[a, :]
    data_label = slabel[a, :]

    a = float(e) / ep
    # l = 2. / (1. + np.exp(-10. * a)) - 1
    lr = 0.01 / (1. + 10 * a) ** 0.75
    l = 1

    _,ll,llc,lld1,lld2 = sess.run([solver,loss, loss_c,loss_d1,loss_d2], feed_dict={xstr:data_tr,xs:data_s,xt:data_t, ys:data_label, keep: 0.8, lam:l,learn :lr})
    acc_test = sess.run(accuracy, feed_dict={xstr: tdata, ys: label, keep: 1})
    acc_train = sess.run(accuracy, feed_dict={xstr: sdata, ys: slabel, keep: 1})
    print('step:%s , train accuracy: %s, test accuracy: %s, ll: %s,llc: %s,lld1: %s,lld2: %s' % (e,acc_train,  acc_test, ll,llc,lld1,lld2))

saver = tf.train.Saver()
save_path=saver.save(sess,"Tensflowsave/Net.ckpt")
print("Save to path:",save_path)












