機械学習エンジニアインターン生の杉崎です。 今回は時系列データ予測に__一次元畳み込み層__を使用した際の出力の可視化の方法について書きたいと思います。

## 本記事の目的
　深層学習における畳込み層は多くは画像等の２次元データに使われることが多いです。そして、ブラックボックスであるモデルに対して理由を明らかにするため、中間層の重みや出力を取り出し可視化する様々な手法が提案されています。(下図)

<img src='https://lh3.googleusercontent.com/QLrUtiwrt6tLrdUaVyDZP_5mFwN7gDKZFxkQKsAbp_BokA8X_pxoxcbA8mu3OOFX81ZZpBe5ai_HjP48yscWq6M=s600'/>
[画像引用元](
https://github.com/raghakot/keras-vis/blob/e019cc43ce6c00b2151941b18dbad63164ad632a/examples/vggnet/activation_maximization.ipynb)

　しかし、そんな中で一次元畳込み層(Conv1d)を用いたモデルでは可視化の事例があまり多くありません。そこで今回はConv1dの重みや中間層の出力の可視化の一例についてご紹介します。

## 環境

- OS : Ubuntu 16.04 LTS
- Python : Python3.5.2
- Jupyter
  - jupyter 4.4.0
  - jupyter-notebook 5.6.0


## サンプル時系列データの作成
　今回使用するサンプル時系列データは正弦波です。

$$
\textbf{toyfunc(t)} = \sin\left( \frac{2\pi}{T} t \right)
$$


この関数を表すコードを以下に示します。

```python
def mySin(t, period=100):
    """
    時刻t, 周期period
    """
    return np.sin(2.0 * np.pi * t/period)

# Add Noise
def toy_func(tStart=0, tStop=100, tStep=1, noiseAmpl=0.05):
    """
    T : sin波の周期
    ampl : ノイズ振幅調整
    """
    t = np.arange(start=tStart, stop=tStop, step=tStep)
    noise = np.random.randn(t.shape[0])  # mean=0, standard_deviation=1
    return (t, mySin(t=t, period=100) + noiseAmpl * noise)
```

<img src='https://lh3.googleusercontent.com/DSpABQV443LT4Z5pFRd3nHXFu6koB-dO4yhqnEP4_tEqYSSgWIkegAk-R96dqhZA-9ob5zAVgzvFBMZgctowqJpR=s750'/>


## 一次元畳み込み層による時系列予測モデル作成
　まず時系列予測を行うConv1dを用いた学習済みモデルが必要なので、サンプルデータとモデルの学習を行います。ソースコードは[こちら](https://github.com/pollenjp/article_script/blob/master/20180719_visualize_Conv1d_output__in_kabuku/notebooks/create_OnlyConv1dModel__SimpleSinFuncWithNoNoise.ipynb)です。

### 入力データの前処理
　次にモデルに入力する形に変えてやります。ここで行おうとしているは過去100個分のデータ(ウィンドウサイズ60)を用いてその一つ先のデータを予測する(予測サイズ1)というものです。以下のGIF動画は波形サイズが500のデータを例として、モデルの学習時に与えるデータを入力データと正解データに分けた学習データを作ります。
(各ブロックの数字はデータのインデックスです。)

<img src='https://lh3.googleusercontent.com/It5NsXZZEqsCd6BgZN0R7Mu2vJI8wFJPQvSP_KE6LTSl3dsdmNRUI4SkgojQ1NsJIqu6A8KTbdQf346dfSA_OKA=s700'/>

<br>
　この処理を行っているコードが以下になります。

```python
>>> #----------------------------------------
>>> # Parameters
>>> #----------------------------------------
>>> windowsize = 100  # 周期と同じくとる
>>> predictsize = 1
>>> sample_data_size = 10000
>>> wave_size = sample_data_size
>>> trainsize = int(10000*0.8)

>>> #----------------------------------------
>>> # Sample Wave
>>> #----------------------------------------
>>> t, sample_wave = toy_func(tStop=sample_data_size, noiseAmpl=0)
>>> print(sample_wave.shape)
(10000,)

>>> #----------------------------------------
>>> # create input data
>>> #----------------------------------------
>>> input_data  = np.empty(shape=(0, windowsize))
>>> output_data = np.empty(shape=(0, predictsize))
>>> print(input_data.shape)
(0, 100)
>>> print(output_data.shape)
(0, 1)
>>> for i in range( sample_wave.shape[0] - (windowsize + predictsize) + 1 ):
        input_data = np.append(arr=input_data,
                               values=sample_wave[i:(i+windowsize)].reshape(1, -1),
                               axis=0)
        output_data = np.append(arr=output_data,
                                values=sample_wave[(i+windowsize):(i+windowsize+predictsize)].reshape(1, -1),
                                axis=0)
>>> print("input_data.shape  : ", input_data.shape)
input_data.shape  :  (9900, 100)
>>> print("output_data.shape : ", output_data.shape)
output_data.shape :  (9900, 1)
>>> #--------------------
>>> # Kerasのモデルに入力できる形にするためにreshapeして次元を足す
>>> #--------------------
>>> input_data = input_data.reshape((-1, windowsize, 1))
>>> output_data = output_data.reshape((-1, predictsize,))
>>> print("input_data.shape  : ", input_data.shape)
input_data.shape  :  (9900, 100, 1)
>>> print("output_data.shape : ", output_data.shape)
output_data.shape :  (9900, 1)
>>> train_x, test_x = input_data[:trainsize], input_data[trainsize:]
>>> train_y, test_y = output_data[:trainsize], output_data[trainsize:]
>>> print("train_x.shape : ", train_x.shape)
train_x.shape :  (8000, 100, 1)
>>> print("train_y.shape : ", train_y.shape)
train_y.shape :  (8000, 1)
>>> print("test_x.shape  : ", test_x.shape)
test_x.shape  :  (1900, 100, 1)
>>> print("test_y.shape  : ", test_y.shape)
test_y.shape  :  (1900, 1)<Paste>
```


### 予測モデル作成 (Keras)
　今回はConv1Dを用いた小さめのモデルを作成します。
　入力や各層の出力の形は`(batch, steps, channels)`のように表されますが、今回は入力のstepsがウィンドウサイズなのでConv1Dの出力stepsもウィンドウサイズと同じ100に設定しています。


```python
>>> from keras.models import Sequential
>>> from keras.layers.convolutional import Conv1D, UpSampling1D
>>> from keras.layers.pooling import GlobalMaxPooling1D

>>> channel_size = 8
>>> kernel_size = 10

>>> model = Sequential()
>>> model.add( Conv1D(filters=channel_size, kernel_size=kernel_size,
                      strides=1, padding="same", activation="relu",
                      input_shape=(windowsize, 1) ) )
>>> model.add( Conv1D(filters=1, kernel_size=8, padding='same', activation='tanh' ) )
>>> model.add( GlobalMaxPooling1D() )

>>> model.compile(loss='mse', optimizer='adam')
>>> model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 100, 8)            88        
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 100, 1)            65        
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 1)                 0         
=================================================================
Total params: 153
Trainable params: 153
Non-trainable params: 0
_________________________________________________________________
```


### モデルの学習
　それではこのモデルを学習して保存します。`model.save()`の`overwrite`を`False`にすると同じ名前のファイルが存在しているとき上書きするかを確認するようになります。

```python
#--------------------
#  Fit
#--------------------
epochs = 100
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]
history = model.fit(x=train_x,
                    y=train_y,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=callbacks_list)
#--------------------
#  Save Model
#--------------------
modelpath = str(keras_model_h5_Path / "model__{}_kernelsize{}.h5".format(ipynb_title, kernel_size))
model.save(filepath=modelpath, overwrite=False)
```


### 学習過程の確認
　下図はepochsを重ねるごとのlossの値です。学習データに対するlossとバリデーションデータに対するlossがともに収束しているため、上手く学習できています。

<img src='https://lh3.googleusercontent.com/WTNQMpqJyBn6wHLQN_L_kUYe1fwYt3Qat9edw97wht07Sol118u10LXFTD5gdVwk8jt8T_JJkJDFyOC_sOD_R3Q=s750'/>



## 中間層Conv1Dの出力を取得
　モデルの学習が終わりましたのでこのモデルで使用したConv1D層のどのチャネルが、波形のどの部分に強く反応しているのかを可視化してみます。
　一連のipynbファイルは<a href="https://github.com/pollenjp/article_script/blob/master/20180719_visualize_Conv1d_output__in_kabuku/notebooks/visualize_OnlyConv1dModel__SimpleSinFuncWithNoNoise.ipynb">こちら</a>にあります。

### モデルの読み込み
　それでは先ほど保存したモデルを読み込みます。

```python
>>> from keras.models import load_model
>>> model_filename = "model__create_OnlyConv1dModel__SimpleSinFuncWithNoNoise_kernelsize10.h5"
>>> modelpath = str(keras_model_h5_Path / model_filename)
>>> model = load_model(filepath=modelpath)
>>> model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 100, 8)            88        
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 100, 1)            65        
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 1)                 0         
=================================================================
Total params: 153
Trainable params: 153
Non-trainable params: 0
_________________________________________________________________
```

### 入力データの作成と処理
　`conv1d_1 (Conv1D)`の出力を得るためには一度データを入力する必要がありますので、学習に使用したものと同様に波形を作成します。学習ではt=0〜9,999のデータを使いましたので、ここで入力するデータはt=10,000〜10,300とします。
　波形サイズ:300、ウィンドウサイズ:100、予測サイズ:1の入力データとします。

```python
tStart = 10000
windowsize = 100
predictsize = 1
wave_size = 300
assert wave_size - windowsize > windowsize  # 後の畳み込むコードではこの条件が必要

>>> #----------------------------------------
>>> # create a wave
>>> #----------------------------------------
>>> t, wave = toy_func(tStart=tStart, tStop=tStart+wave_size, noiseAmpl=0)

>>> #----------------------------------------
>>> # create input data
>>> #----------------------------------------
>>> input_arr = np.empty(shape=(0, windowsize))
>>> print(input_arr.shape)
(0, 100)
>>> for i in range( wave.shape[0] - (windowsize + predictsize) + 1 ):
>>>     input_arr = np.append(arr=input_arr,
>>>                            values=wave[i:(i+windowsize)].reshape(1, -1),
>>>                            axis=0)
>>> print("input_arr.shape  : ", input_arr.shape)
input_arr.shape  :  (200, 100)
>>> input_arr = input_arr.reshape((-1, windowsize, 1))
>>> print("input_arr.shape  : ", input_arr.shape)
input_arr.shape  :  (200, 100, 1)
```


### 中間層の出力の取得
　作成したデータを入力して`conv1d_1 (Conv1D)`から出てきた出力値を取得します。

```python
from keras.models import Model
output_layer_name = 'conv1d_1'
hidden_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer(output_layer_name).output)

hidden_layer_output = hidden_layer_model.predict(x=input_arr)
print(hidden_layer_output.shape)  # 出力: (440, 60, 60)
```

　出力結果のサイズが`(440, 60, 60)`であることとsummaryで確認したconv1d_1 (Conv1D)のOutput Shape`(None, 60, 60)`の下２つのサイズが一致していることを確認してください。これでconv1d_1 (Conv1D)の出力を取得出来たことがわかります。


### Conv1Dの出力の描画
　conv1d_1 (Conv1D)の出力が取得出来たので、このデータをカラーマップで描画してみます。しかし、`(440, 60, 60)`のサイズの結果が得られたので出力するには２次元に落としこむ必要があります。そこでこのサイズの意味を考えてみます。
　畳み込みにおいて、入力波形サイズを`(440, 60, 1)=(N, windowsize, 1)`とすれば、出力波形サイズは`(440, 60, 60)=(N, windowsize, filter_size)`となって出てきます。つまり、__N__と__windowsize__は保存されます。ここでは__filter_size__を__channel__と読み替えたいと思います。そこで一つのchannelの出力を取り出すことで一つの入力サイズと同じ大きさの値を取り出せます。
　ここでは試しに最初のchannel (index 0)と取り出して




## 参考

- <a href="https://qiita.com/icoxfog417/items/8689f943fd1225e24358">ディープラーニングの判断根拠を理解する手法 - Qiita</a>
- <a href="https://qiita.com/niisan-tokyo/items/a94dbd3134219f19cab1">時系列予測を一次元畳み込みを使って解く with Keras - Qiita</a>
- <a href="http://roomba.hatenablog.com/entry/2017/04/21/154954">TensorFlowでのSoftmax回帰の実装・可視化・識別器の騙し方 - roombaの日記</a>
- <a href="https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-visualizing-convolutional-neural-networks/">Essentials of Deep Learning: Visualizing Convolutional Neural Networks in Python</a>



