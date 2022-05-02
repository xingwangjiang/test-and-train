from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据集  https://blog.csdn.net/qq_37289115/article/details/108769974?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165105235416782246461828%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165105235416782246461828&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-108769974.142^v9^control,157^v4^control&utm_term=%E9%B8%A2%E5%B0%BE%E8%8A%B1&spm=1018.2226.3001.4187
iris = load_iris()
# 共有150个例子， 数据的类型是numpy.ndarray
print(iris.data.shape)#(150,4)
# 对应的标签有0,1,2三种
print(iris.target.shape)#(150,)
# 使用TSNE进行降维处理。从4维降至2维。
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(iris.data)
                                                               #n_components：int，可选（默认值：2）嵌入式空间的维度。降维后的维度是多少learning_rate：float，可选（默认值：1000）学习率可以是一个关键参数。它应该在100到1000之间。如果在初始优化期间成本函数增加，则早期夸大因子或学习率可能太高。如果成本函数陷入局部最小的最小值，则学习速率有时会有所帮助。
                                                               #fit_transform	将 X 投影到一个嵌入空间并返回转换结果
                                                               #https://blog.csdn.net/weixin_44530236/article/details/89309046?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165105924316782425132931%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165105924316782425132931&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-89309046.142^v9^control,157^v4^control&utm_term=tsne+%3D+TSNE%28n_components%3D2%2C+learning_rate%3D100%29.fit_transform%28iris.data%29&spm=1018.2226.3001.4187


# 使用PCA 进行降维处理
pca = PCA().fit_transform(iris.data)



# 设置画布的大小                                         #https://blog.csdn.net7baoziqyp/article/details/111239877?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.figure(figsize&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-111239877.142^v9^pc_search_result_control_group,157^v4^control&spm=1018.2226.3001.4187
plt.figure(figsize=(12, 6))                             #figsize:指定figure的宽和高，单位为英寸,图片的像素（右上角的PNG）
plt.subplot(121)                                        #plt.subplot(232)将figure分成2*3=6个子图区域，第三个参数2表示将生成的图画在第2个位置
plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)      #对于X[:,0];是取二维数组中第一维的所有数据，对于X[:,1]是取二维数组中第二维的所有数据，对于X[:,m:n]是取二维数组中第m维到第n-1维的所有数据  https://blog.csdn.net/Together_CZ/article/details/79593952?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165105167616782391844606%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165105167616782391844606&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-79593952.142^v9^control,157^v4^control&utm_term=%5B%3A%2C+0%5D&spm=1018.2226.3001.4187
                                                        #c：表示的是色彩或颜色序列，可选，默认蓝色’b’。但是c不应该是一个单一的RGB数字，也不应该是一个RGBA的序列，因为不便区分。c可以是一个RGB或RGBA二维行数组。
plt.subplot(122)
plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
plt.colorbar()
plt.show()



