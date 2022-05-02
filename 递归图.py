from pyts.image import RecurrencePlot
import numpy as np
X=np.random.random((1,1024,1024))
y=np.random.random((300,1024))
for i in range(1,300,1):
    x = y[i,:]
    x=np.reshape(x,(1,-1))
    #print(x.shape)
    rp =RecurrencePlot(threshold='point',percentage=20)
    x_rp = rp.fit_transform(x)
    X = np.concatenate((X,x_rp),axis=0)
    print(X.shape)