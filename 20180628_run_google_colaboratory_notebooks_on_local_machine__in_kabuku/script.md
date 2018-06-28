

<!--
title:
Google Colaboratory を自分のマシンで走らせる
-->

　株式会社カブクで機械学習エンジニアのインターンでお世話になっている杉崎弘明（大学3年）です。今回はGoogle Colaboratoryのローカル実行について書きます。

# 本記事の目的
　Google Colaboratory（以降、Colaboratory）というサービスをご存知でしょうか。このサービスはGoogle Driveなどを通じてJupyter Notebooksのような環境を管理することができるサービスになります。iPython Notebook（以降ipynb）上のセルを実行するとデフォルトでGoogleが提供してくれているサーバー上で動くことになりますが、今回は手元のPCやリモートサーバ上でColaboratoryを実行していきたいとおもいます。
　ipynbをすぐに共有できるGoogle Colaboratoryは魅力的だけどローカルにあるCPUやGPUパワーを使用したいといった際に活用できます。
　基本的な手法はGoogle公式のページに掲載されています。https://research.google.com/colaboratory/local-runtimes.html?hl=ja

# 実行環境
今回使用したマシンは以下のような構成になっています。

- Ubuntu 16.04 LTS (GPUサーバ)
- Anaconda
	- Python3.5.2
	- TensorFlow 1.8.0 (CPU、GPUバージョンの比較あり)
	- Keras 2.2.0
- ブラウザ : ChromeまたはFirefox



# 目次
1. ローカルのラップトップPC(Mac)でGoogle Colaboratoryを実行
2. ローカルのGPU搭載デスクトップPC (Ubuntu) でGoogle Colaboratoryを実行
3. GPUマシンにリモートデスクトップで接続する方法
	3.1. ディスプレイ有りGPUマシンをミラーリングで操作
	3.2. ディスプレイ無しGPUマシン（ヘッドレスサーバ）を仮想デスクトップ環境下で操作
4. おまけ1：AnacondaによるPython環境の構築(Ubuntu)
5. おまけ2：【失敗】SSHポートフォーワディングを利用したJupyterとColaboratoryの接続



# 1. ローカルのラップトップPC(Mac)でGoogle Colaboratoryを実行
　やり方はとっても簡単です。基本的には以下の４ステップです。

1.1. Jupyterを使用したいPCにインストールする
1.2. Colaboratoryが接続できるようになる拡張機能`jupyter_http_over_ws`をいれる
1.3. Jupyterを起動する
1.4. Colaboratoryを開いてローカルランタイムを選択する

詳しい環境構築を見たい方は「4. おまけ1：AnacondaによるPython環境の構築(Ubuntu)」をご覧ください。


## 1.4. Jupyterを使用したいPCにインストールする。
Python環境が構築された状態でpip(pip3)コマンドを使用してJupyterをインストールします

```
$ pip install jupyter
```

## 1.2. Colaboratoryが接続できるようになる拡張機能`jupyter_http_over_ws`をいれる
Jupyterの拡張機能<a href="https://github.com/googlecolab/jupyter_http_over_ws">jupyter_http_over_ws</a>を導入し有効にします。

```
$ pip install jupyter_http_over_ws
$ jupyter serverextension enable --py jupyter_http_over_ws
```

## 1.3. Jupyterを起動する
Jupyterを起動する際にアクセスを通すURLとポートを指定してあげます。
ポートは8888に限らず空いているポートを自由に指定できます。

```
$ jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888
```

このコマンド実行後、デフォルトのブラウザが起動します。

## 1.4. Colaboratoryを開いてローカルランタイムを選択する
　先ほどデフォルトで開いたブラウザと同じブラウザを用いてColaboratoryのノートブックを開きます。
　右上にある「connect」と表示されているところをクリックし、「Connect to a local runtime...」（「ローカル ランタイムに接続」）を選択します。

<img src='https://lh3.googleusercontent.com/LufXzkxgpP2QSntsrBLmMAP0FBLGopSR7rclDc6f_e8snugRTC0PnB9VWSSfWjLyfBjOxkCH3IH2j-VYPQGh--k=s600'/>
<img src='https://lh3.googleusercontent.com/Vi2v97lM3jeVsq2oqMcz_rCTrwP08x2IQkML4D5dRP5emsz6-7qRBPdC5ufgZZO0lV8ZX0lD2Dz_U2Nja1w4uPY=s600'/>


ポート番号が先ほど示したものと同じであることを確認して「CONNECT」をクリックします。
<img src='https://lh3.googleusercontent.com/UgUJVf_m7oRUzl5hpLnYf_jR-F196F3NBksOfTEir9Uo9wTKon-74KJ1mgHUwCjUXrXJ0rcvXr99oyWEBKcOmkxJ=s600'/>


### 1.4.1. Firefoxで使用する場合
　Chromeだと素直につながりますがFirefoxだとデフォルト設定で弾かれるようになっているので以下のように設定してください。
　まずFirefoxブラウザのアドレスバーに「about:config」と入力してEnterを押してください。警告画面がでますので「危険性を承知の上で使用する」をクリックします。
<img src='https://lh3.googleusercontent.com/6ct_rJs_Lw0iQfuUQ_CoUpTOkYg4woOOXkbp2uDp80xn770oO0B0JLeYksB8qIuiZWZBr4gbbpy4rJdjTG4DdN4=s600'/>


検索欄に「network.websocket.allowInsecureFromHTTPS」と入力して出てきた項目をダブルクリックで「true」に変更します。
<img src='https://lh3.googleusercontent.com/8DR4i28RUhhj681KnJ5ue_qagnSfPo6-Lx9sm22EA7xJKaaD7EdN9ygEZ_YbbLcs0PxprbmXUFeogNkx7EhxPg=s600'/>

　これでFirefoxからのアクセスも通るようになります。


# 2. ローカルのGPU搭載デスクトップPC (Ubuntu) でGoogle Colaboratoryを実行
　次はGPU搭載のデスクトップPCを使用したいと思います。



## 2.1. GPUマシンでのColaboratoryの実行
　TensorFlowやKerasで手元のGPUを使いたい場合はColaboratory側でいじる設定は無く、ローカルのマシンにインストールされているTensorFlowに依ります。CPU用のTensorFlowをインストールして入ればCPUが使われますし、GPU用のTensorFlowをインストールして入ればGPUが使われます。

　CPUとGPUの速度の違いを以下に記しておきます。
　実行したコードはKeras-teamが公開している<a href="https://github.com/keras-team/keras/blob/ebdc1c8759f65768212b7e7113b5cae82e9df3d4/examples/cifar10_cnn.py">cifar10のサンプルコード</a>になります。実際にタイムを測定したのは以下の部分にです。
(CPUでの実行が遅すぎたのでサンプルコードに一部変更を加え`epochs`を5で実行しています。)

```python
%%time

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
```

の比較を行うと以下のようになります。

|     | CPU | GPU |
|:---:|:---:|:---:|
| 実行時間 (Wall Time) | 6min 57s | 56.3 s |
| 識別精度 (Test Accuracy) | 0.569 | 0.5803 |

　5エポックしかまわしていないので識別精度はそれほど高くないですが、ほぼ７倍の速度で同じレベルの識別精度を達成しています。これでちゃんとGPUが使われていることが確認できました。


# 3. GPUマシンにリモートデスクトップで接続する方法
　GPUマシンを直接扱う人以外にリモートで扱いたい人もいると思いますので以下にその一手法をまとめておきます。
- 3.1. ディスプレイ有りGPUマシンをミラーリングで操作
- 3.2. ディスプレイ無しGPUマシン（ヘッドレスサーバ）を仮想デスクトップ環境下で操作

※リモートデスクトップで実行しようとしている理由はSSHポートフォワーディングによるJupyterとColaboratoryの接続がうまくいかなかったからです。（後述：「おまけ2：【失敗】SSHポートフォーワディングを利用したColaboratoryの実行」）


## 3.1. ディスプレイ有りGPUマシンをミラーリングで操作

### 3.1.1. Ubuntuサーバ側の設定
　まず左上のアイコンをクリックして検索ワードに「desktop sharing」と入力します。すると、「Desktop Sharing」アプリケーションが見つかりますのでクリックします。
<img src='https://lh3.googleusercontent.com/wK2DF8ugInIAskSs_Olc_UIcDZnpSLgC6oCRH0Jx_T7WqILUkzE07gJm8WL8rQkcek3pja-lJoOWjdIhgsW1GnQk=s600'/>

　以下のように設定してパスワードを入力し、閉じます。
<img src='https://lh3.googleusercontent.com/f-JV_yGvJRlhjPNDrT8R7UsMigmJFyXmMmaQDKZfZhkuBemmI3iU_lqNTSG0Bcl6NofNAF-aUfNkOZmmcg7N3Q=s600'/>

　次に暗号化をオフにしておかないとMacからの接続ができないので以下のコマンドをターミナルで実行します。
```
$ gsettings set org.gnome.Vino require-encryption false
```

ここまで終わったら一度パソコンを再起動し、再びログインしてください。これでUbuntu側の設定は終わりです。


### 3.1.2. Mac（クライアント）側の設定
　MacからSSHでポートフォーワディング接続をして、5900番ポートをつなぎます。ここではMac側のポートを55900にしていますが、空いていれば何でも構いません。
```
$ ssh -L localhost:55900:localhost:5900 <user>@<ip>
```
　この状態のまま手元のMacで「Finder」を開きます（Cmd + Space, "finder"を入力, Enter）。そして写真のように「Connect to Server」をクリックまたは「Cmd + k」を押します。

<img src='https://lh3.googleusercontent.com/dXDCdyAI4ltKLTbzgQMuK_MdceyNdl6jAfUv7BehHwL2w_h0HoOtqhey-7GYnFsvdW63Ni2jNxrhnTszO9snGA'>

　今回の場合、アドレス欄に「vnc://localhost:55900」と入力し、Connectをクリックします。
<img src='https://lh3.googleusercontent.com/7cUmLG7X7tSIOSn9NUtUiEaQfdXBMeqlZ06hQ4axXlpNUUeoZ-6PqLzQHEX7-Sh2Xadu3RhQn_6y1GfRJU1bAQ=s600'/>

パスワードが求められますので先に決めたパスワードを入力してあげればUbuntu側のディスプレイに映っている画面と同じ画面を操作することができます。
<img src='https://lh3.googleusercontent.com/yFuXQwEV08ReaU0KUlkII63uMPgmDRMPhoyvtWUMYOMAHKI3HJiPfrXmR24x5nPYXQvhO7P2EyNjGbtKoOOIkA=s600'/>

<img src='https://lh3.googleusercontent.com/_L4iXBRQm8EjV3w0T7elfHNlb16Bfox3RbFgVuuyTF0aUt2aJkL1DdiLDMJw_iti4USIRn-riJteenzsbI2KnQ=s600'/>


　仮想的なデスクトップである分、ミラーリングのときよりもサクサク動いた印象を受けます。


## 3.2. ディスプレイ無しGPUマシン（ヘッドレスサーバ）を仮想デスクトップ環境下で操作
　次にディスプレイをつけていないヘッドレスサーバにおけるリモートデスクトップの方法を記述します。tightvncserverを利用します。Ubuntuにはデフォルトで入っているかもしれませんが一応インストール方法も含めて説明します。

### 3.2.1. サーバ側
　SSHなどを経由してサーバにログインして設定を行います。
　まずは必要なパッケージをインストールします。
```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get dist-upgrade
$ sudo apt-get install xorg lxde-core
$ sudo apt-get install ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal
$ sudo apt-get install tightvncserver
```

　次に`tightvncserver`コマンドを利用してVNCサーバを立ち上げます。初めて起動させると自動で`~/.vnc`ディレクトリ下にファイルを生成してくれるので一度以下のコマンドを実行します。
```
$ tightvncserver -localhost :1
```
　このときリモートデスクトップする際に用いるパスワードを聞かれますので入力しましょう。

　そして以下のコマンドを実行して終了します。
```
$ tightvncserver -kill :1
```

　さて、`~/.vnc/xstartup`ファイルを以下のように書き換えます。既存のものはコメントアウトして以下の内容を追記してください。
```
#!/bin/sh~

[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup~
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources~
xsetroot -solid grey~
vncconfig -iconic &~
x-terminal-emulator -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop" &~
x-window-manager &~
~
gnome-panel &~
gnome-settings-daemon &~
metacity &~
nautilus &~
```

　最後にこの設定の状態で再び起動します。
```
$ tightvncserver -localhost :1
```

このとき、5900番から始めて`:1`の場合は1番目、つまり5901番ポートが空いていることを確認しましょう。
```sh
$ netstat -plan | grep Xtightvnc
tcp        0      0 127.0.0.1:5901   0.0.0.0:*       LISTEN      11986/Xtightvnc 
tcp        0      0 0.0.0.0:6001     0.0.0.0:*       LISTEN      11986/Xtightvnc 
```

　これでサーバ側の設定は終わりです。


### 3.2.2. クライアント(Mac)側
　　「3.1.2. Mac（クライアント）側の設定」と同様、SSHポートフォワーディングした上でfinderから接続します。
```
$ ssh -L locahost:55901:localhost:5901 <user>@<ip>
```

<img src='https://lh3.googleusercontent.com/wXeEY5CSAo7xqswkwOo8EfpYJyD23xK_TdBiXvxmrBLWvFvN0P88p_jXJsEiv3zvvndyKFLRn31EYn-QKN89yg=s600'/>


<img src='https://lh3.googleusercontent.com/mIaq2GCH2JYF6usH1Sz5ipfoq4Hw6-PuvuPAg1hd6MbTv0S_kncg8zYltWKRwAsXun6T_xaDgoyNR-koUFjMfA=s600'/>

　左上の「アプリケーション > システムツール > Xfce Terminal」を選択すればターミナルを開けます。
<img src='https://lh3.googleusercontent.com/RveG9YamF2bpkPcGl-H839VQR7T3mnP_LQuWSdAdpzKzA3bhmVHFSDBmKZiZU9eXdFg-GhEgfNRzBf5j_TjrHQ=s600'/>

<img src='https://lh3.googleusercontent.com/9nhqWbGlM9r5qAnBfEfRdHkvApT6lTZaqHD29moFw2ArAniw9SMzdcQeFpTDcSWEOigc5x3XYsth2xfrljY1KmLd=s600'/>

　以下のようにGPIからブラウザを起動することもできます。
<img src='https://lh3.googleusercontent.com/BNbSK794sdx68wIakQiHrwtsezt2b3aQdYMSXB4oDJXPS0zcHzu9SI5TsthDxRQYVOIAIa6JG40dzaRreHef3LQ=s600'/>

　これ以降は通常のデスクトップと同様なので、「1. ローカルのラップトップPC(Mac)でGoogle Colaboratoryを実行」で説明した内容と同じ方法でJupyterを起動しColaboratoryと接続してください。


# 4. おまけ1：AnacondaによるPython環境の構築(Ubuntu)
　今回使用した環境の構築方法を掲載しておきます。

## 4.1. Anacondaのインストール
　まず下のサイトで自分のインストール環境のOSを選択します。ここではUbuntuへの環境構築なのでLinuxを選択します。
https://www.anaconda.com/download/
<img src='https://lh3.googleusercontent.com/pRCBsYtqbOcgPPXGS3hwHIQF9LRbz-3oL7klhGAXECRu2_HukRDuWvSzRq6bKxoxQvwm9YyW7tUX_d4DLdbFdg=s600'/>

Download下の「64-Bit (x86) Installer」を右クリックしてリンクのURLをコピーします。
<img src='https://lh3.googleusercontent.com/JlOZo8zmKHd3ozvZyrQOBMJRMQIoOD72spAvzo7aulRr2DRBDl0gZROfcYrbYdzXLJsNPx2R2hSORuztLqJQglE=s600'/>

サーバー環境のターミナルに入り以下のコマンドを実行します。
（ここではコピーしたURLを `https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh` だとします。）

```
$ cd ~
$ wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
$ bash Anaconda3-5.2.0-Linux-x86_64.sh
```

公式のページは<a href="https://conda.io/docs/user-guide/install/linux.html">こちら</a>であり、そこには「設定に関してわからなければデフォルトで良い」と書いてあるので、デフォルトのままで大丈夫です。

　インストールが完了したら以下のコマンドで`python3.5.2`の環境を作ります。
```
$ cd ~
$ source .bashrc
$ conda create --name py3.5.2 python=3.5.2
```

さらに以下のコマンドを利用すれば`python3.5.2`の環境下に入ることができます。

```
$ source activate py3.5.2
```



## 4.2. TensorFlow / Keras のインストール
　<a target="_blank" href="https://www.tensorflow.org/install/install_mac#installing_with_anaconda">公式</a>に書いてある通りですが、<a target="_blank" href="https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package">こちら</a>からバージョンに合うもののURLをコピーしてpipコマンドでインストールします。
```sh
$ source activate py3.5.2
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
```

kerasは単にpipでインストールできます。
```sh
$ pip install keras
```

以上で環境構築は完了です。



# 5. おまけ2：【失敗】SSHポートフォーワディングを利用したJupyterとColaboratoryの接続
　リモートのGPUマシン上で起動したJupyterをSSHポートフォワーディングを使用して、ローカルブラウザ上のColaboratoryと接続を試みたところエラーが出てうまく接続できなかったので記述しておきます。

　例えば以下のような図で表される感じをイメージしています。

<img src='https://lh3.googleusercontent.com/B3kdVv4atqWMhnmj73DwXXC_uEyrO4eFkXGeRSbniW-K5EHRUszHLbh2gH7AGkYFBoHjcmlrC3k3xUEPxQdMv9k=s600'/>

　コードで書くと以下のようになります。

```
pc@local:$ ssh -L localhost:58888:localhost:8888 <user>@<ip>
<user>@server:$ source activate py3.5.2
<user>@server:$ jupyter notebook \
  	--NotebookApp.allow_origin='https://colab.research.google.com' \
  	--port=8888 ¥
  	--no-browser
```

　最後のコマンド実行時にログにtoken付きのlocalhostのURLが表示されるのでそれをコピーします。lcoalのブラウザに貼り付けた状態でポートを58888に変更しますとちゃんとJupyterへのアクセスが可能になります。

しかし、同じブラウザを使用してColaboratoryの接続ポートを58888に設定しても接続されませんでした。
<img src='https://lh3.googleusercontent.com/r9H6mTAx5I9mcVZWu7ewmlRgp2m8yJOdOVA4LEAVsHEaGiOrUHt85TpUa3YZrqRc3BmBBRX_MXH3WiHM0Dcykw=s600'/>


ずっと「接続中」のままで接続されません。
<img src='https://lh3.googleusercontent.com/ikxcyRjjrvVxQLsYtt1QpH3ifO3L68ziO5WztqG6wEhYU-D4wJpDNS3qGvh4lA9nIlxGxH9_XNBazlU-nbKKqCc=s600'/>

その際、Jupyterのログには以下のようなコードが出力されていました。

```
[E 19:21:06.168 NotebookApp] Uncaught error when proxying request
    Traceback (most recent call last):
      File "/home/<user>/anaconda3/envs/py3.5.2/lib/python3.5/site-packages/jupyter_http_over_ws/handlers.py", line 151, in on_message
        response.rethrow()
      File "/home/<user>/anaconda3/envs/py3.5.2/lib/python3.5/site-packages/tornado/httpclient.py", line 611, in rethrow
        raise self.error
    ConnectionRefusedError: [Errno 111] Connection refused
```

拡張機能`jupyter_http_over_ws`の対応を待った方が良さそうです。


# 最後に
　今回はGoogle Colaboratoryのipynbをローカルマシンで実行する方法と、リモートサーバでも扱えるようにリモートデスクトップの方法についても記述しました。アップデートを通じてリモートデスクトップの必要はなくなるかもしれませんが、それまでの間参考にしていただけると幸いです。




# 参考文献
- Google Colaboratory
	- <a target="_blank" href="https://research.google.com/colaboratory/local-runtimes.html?hl=ja">ローカル ランタイム - Google</a>
	- <a target="_blank" href="https://github.com/keras-team/keras/blob/ebdc1c8759f65768212b7e7113b5cae82e9df3d4/examples/cifar10_cnn.py"> keras/examples/cifar10_cnn.py keras-team/keras - GitHub</a>
	- <a target="_blank" href="https://conda.io/docs/user-guide/install/linux.html">Installing on Linux - Conda</a>
- リモートデスクトップ
	- Vino
		- <a target="_blank" href="https://blog.mosuke.tech/entry/2015/08/13/000440/">デスクトップUbuntuにVNC接続。ついでにSSHローカルポートフォワードの復習。</a>
	- tightvncserver
		- <a target="_blank" href="https://help.ubuntu.com/community/VNC/Servers#tightvncserver">tightvncserver VNC/Servers - Community Help Wiki - Ubuntu Documentation</a>
		- <a target="_blank" href="https://knowledgelayer.softlayer.com/learning/tightvnc-server-ubuntu-1604">TightVNC Server on Ubuntu 16.04</a>
		- <a target="_blank" href="https://www.linode.com/docs/applications/remote-desktop/install-vnc-on-ubuntu-16-04/">Install VNC on Ubuntu 16.04</a>
		- <a target="_blank" href="vncserver grey screen ubuntu 16.04 LTS">vnc - vncserver grey screen ubuntu 16.04 LTS - Ask Ubuntu</a>



