
Proje Raporu >> YSA_ProjeFinalRaporu.pdf


Kodlar çalıştırılırken öncelikle “Preprocessing.py” scripti çalıştırılır ve veri seti  eğitim için hazır hale gelir ve kaydedilir.

“PricePredictionWithKerasMLP.py” ve “PricePredictionWithNumpyMLP.py”, “Preprocessing.py” ile hazırlanıp kaydedilen verisetini içeri aktarır ve öyle çalışır.

Not: Numpy MLP’de tüm veri seti için eğitim uzun sürmektedir. Bunun için verisetinden sayılı örnek alarak çalıştırılabilir.
listind=np.linspace(start=0,stop=20427,num=3000).astype(int) 
df = pd.read_pickle("df.pkl").iloc[listind,:] #3000 örnek alınarak bu şekilde çalıştırılabilir.

Not: Numpy MLP’de Dropout katmanının oranı, Train metodu ve Predict metodu çağrılırken girdi olarak verilir. Train ve Predict için dropout oranları aynı olmalıdır.
Ağ2’ı dropoutsuz çalıştırmak için Dropout oranları sıfır olarak belirlenir.
