import numpy as np

class MultiP():
    def __init__(self,inputDim=50,firstLayer=20,secondLayer=10,outputDim=4):

        self.inputDim=inputDim; self.firstLayer=firstLayer; self.secondLayer=secondLayer; self.outputDim=outputDim
        # Çok Katmanlı Algılayıcımız iki gizli katmandan oluşuyor. Giriş, çıkış ve katmanlardaki nöron sayıları...
        # Class'ı çağırırken girdi olarak verilebiliyor.

        self.w1=0.6*np.random.rand(firstLayer,inputDim+1)-0.3
        self.w2=0.6*np.random.rand(secondLayer,firstLayer+1)-0.3
        self.w3=0.6*np.random.rand(outputDim,secondLayer+1)-0.3     # Rastgele sayılar ile ağırlıklar oluşturuldu.

    def feedForward(self,X,dropout1=0.3,dropout2=0.3):

        if dropout1 != 0 and dropout2 != 0:
            deactivated1=np.random.choice(self.firstLayer,size=int(dropout1*self.firstLayer),replace=False)
            deactivated2=np.random.choice(self.secondLayer,size=int(dropout2*self.secondLayer),replace=False)
            temp1=self.w1[deactivated1,:]; temp2=self.w2[:,deactivated1]
            temp3=self.w2[deactivated2,:]; temp4=self.w3[:,deactivated2]
            self.w1[deactivated1,:], self.w2[:,deactivated1] = 0,0
            self.w2[deactivated2,:], self.w3[:,deactivated2] = 0,0


        self.X=X
        X=X.reshape(-1,1)   # girişimiz sütun vektörüne çevrildi.
        bias_0=np.ones((X.shape[1],1))
        X_bias=np.concatenate([X,bias_0],axis=0)    # bias eklendi
        self.X_bias=X_bias
        v1=np.dot(self.w1,X_bias)    # ilk katman çıkışı
        self.v1=v1
        y1= self.act(v1,"relu")        # ilk katman çıkışı sigmoid'ten geçirildi ve ikinci katman girişi oluşturuldu.
        self.y1=y1
        bias_1=np.ones((y1.shape[1],1))
        y1_bias=np.concatenate([y1,bias_1],axis=0)      # bias eklendi.
        self.y1_bias = y1_bias
        v2=np.dot(self.w2,y1_bias) #* (1-dropout1)       # ikinci katman çıkışı
        self.v2=v2
        y2=self.act(v2,"relu")             # ikinci katman çıkışı sigmoid'ten geçirildi ve üçüncü katman girişi oluşturuldu.
        self.y2=y2
        bias_2=np.ones((y2.shape[1],1))
        y2_bias=np.concatenate([y2,bias_2],axis=0)      # bias eklendi.
        self.y2_bias=y2_bias
        v3=np.dot(self.w3,y2_bias) #* (1-dropout2)     # üçüncü katman çıkışı
        self.v3=v3
        y3=self.act(v3,"sigmoid")             # üçüncü katman çıkışı sigmoid'ten geçirildi be nihai çıkış elde edildi.
        self.y3=y3
        if dropout1!=0 and dropout2!=0:
            self.w1[deactivated1, :], self.w2[:, deactivated1] = temp1, temp2
            self.w2[deactivated2, :], self.w3[:, deactivated2] = temp3, temp4


        return y3

    def predict(self,X,dropout1,dropout2):


        self.X=X
        X=X.reshape(-1,1)   # girişimiz sütun vektörüne çevrildi.
        bias_0=np.ones((X.shape[1],1))
        X_bias=np.concatenate([X,bias_0],axis=0)    # bias eklendi
        self.X_bias=X_bias
        v1=np.dot(self.w1,X_bias)    # ilk katman çıkışı
        self.v1=v1
        y1= self.act(v1,"relu")        # ilk katman çıkışı sigmoid'ten geçirildi ve ikinci katman girişi oluşturuldu.
        self.y1=y1
        bias_1=np.ones((y1.shape[1],1))
        y1_bias=np.concatenate([y1,bias_1],axis=0)      # bias eklendi.
        self.y1_bias = y1_bias
        v2=np.dot(self.w2,y1_bias) * (1-dropout1)       # ikinci katman çıkışı
        self.v2=v2
        y2=self.act(v2,"relu")             # ikinci katman çıkışı sigmoid'ten geçirildi ve üçüncü katman girişi oluşturuldu.
        self.y2=y2
        bias_2=np.ones((y2.shape[1],1))
        y2_bias=np.concatenate([y2,bias_2],axis=0)      # bias eklendi.
        self.y2_bias=y2_bias
        v3=np.dot(self.w3,y2_bias) * (1-dropout2)     # üçüncü katman çıkışı
        self.v3=v3
        y3=self.act(v3,"sigmoid")             # üçüncü katman çıkışı sigmoid'ten geçirildi be nihai çıkış elde edildi.
        self.y3=y3
        return y3

    def backProp(self,error):

        grad_3=error*self.sigmoid_derivative(self.y3,"sigmoid")       #çıkış katmanı yerel gradyeni
        self.grad_3 = grad_3
        w3=(self.w3.T)[:-1,:]       # gradyen hesabı için kullanılacak ağırlık matrisinde bias satırı çıkarıldı ve transpoze edildi.
        grad_2=np.dot(w3,grad_3)* self.sigmoid_derivative(self.y2,"relu")      #çıkış'tan önceki katmanın yerel gradyeni
        self.grad_2 = grad_2
        w2 = (self.w2.T)[:-1, :]    # gradyen hesabı için kullanılacak ağırlık matrisinde bias satırı çıkarıldı ve transpoze edildi.
        grad_1 = np.dot(w2, grad_2) * self.sigmoid_derivative(self.y1,"relu")      # giriş yerel gradyeni
        self.grad_1=grad_1

    def gradDescent(self,lr):
        self.lr=lr

        self.w3 += lr * np.dot(self.grad_3,self.y2_bias.T)
        self.w2 += lr * np.dot(self.grad_2, self.y1_bias.T)
        x=self.X_bias.reshape(1,-1)                   # yerel gradyenler, ilgili katmanın girişleri ve öğrenme hızı kullanılarak
        self.w1 += lr * np.dot(self.grad_1, x)        # ağırlıklar güncellendi.

    def train(self,X,y,epochs,lr,dropout1,dropout2):

        for i in range(epochs):
            toplam_error = 0

            for j, inp  in enumerate(X):

                target=y[j].reshape(-1,1)  # arzu edilen çıkış
                out= self.feedForward(inp,dropout1,dropout2) #ileri yolda gidilerek bulunan çıkış

                error=target-out        #geri yayılım için kullanılacak olan hata

                self.backProp(error)        # geri yayılım ile yerel gradyen hesapları
                self.gradDescent(lr)       # grad Derscent ile ağırlık güncelleme

                toplam_error += self.meanSE(target, out)
            ort_kareHata = toplam_error / X.shape[0]
            if((i+1)%10==0):
                print("Eğitim için Ortalama Kare Hata: {}, iterasyon sayısı: {}".format(ort_kareHata, i + 1))
                # her 10 iterasyonda bir hatamız yazdırılıyor.
            if(ort_kareHata<0.00001):
                break

    def act(self, x, act_fonc):
        self.act_fonc=act_fonc
        if self.act_fonc=="sigmoid":
            y = 1.0 / (1 + np.exp(-x))
        elif self.act_fonc =="tanh" :
            y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        elif self.act_fonc=="relu":
            y = np.maximum(x,0)
        return y

    def sigmoid_derivative(self, x, act_dx):
        self.act_dx=act_dx
        if self.act_dx=="sigmoid":
            y = x * (1.0 - x)
        elif self.act_dx =="tanh":
            y= 1 - x**2
        elif self.act_dx=="relu":
            y=1
        return y

    def meanSE(self, hedef, cıkıs):
        return np.average(0.5*(hedef - cıkıs) ** 2)
