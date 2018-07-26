株式会社カブクで、機械学習エンジニアとしてインターンシップをしている杉崎弘明（大学3年）です。


## 目次
- 本記事の目的
- 探索的データ解析（EDA）とは何か
- KaggleのコンペティションでEDA
- サイズの大きいデータの扱い方
- DASK
- EDAの実行
- 最後に


<!--
----------------------------------------
-->
## 本記事の目的
本記事では探索的データ解析（EDA）の説明と並列処理フレームワークDASKの処理解説、DASKを用いた実際のデータに対するEDAの一例を紹介いたします。
データはKaggle Competitionにおいて公開されている「<a href="https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection">TalkingData AdTracking Fraud Detection Challenge</a>」を使用します。
Kaggleという言葉を初めて聞いた方は以下のサイトなどをご覧ください。
- https://www.codexa.net/what-is-kaggle/

実行環境
- OS: Ubuntu 16.04 LTS
- メモリ(RAM)サイズ: 8 GB
- 言語: Python3.5.2


<!--
----------------------------------------
-->
## 探索的データ解析（EDA）とは何か
探索的データ解析（Exploratory Data Analysis: EDA）は、John W. Tukeyによって提唱された考え方であり、データが持っている様々な側面の情報から特徴を取り出すアプローチ全般のことです。John W. Tukeyは当時、仮定の上でのみ成り立つ数理的統計だけでなく、実際のデータの解析を重要視し、探索的データ解析として箱ひげ図などの可視化によるアプローチを開発しました。
探索的アプローチは複雑なデータからモデルが適用できるような特徴を見つけることに意味があります。例えば、現実のデータは大変複雑な構造をしているので数理統計によるモデルの仮定を最初から満たしてくれません。そこでデータの特徴を上手く把握することでで、それに応じたモデルの選択が可能になります。

EDAの方針としては以下のようなものが挙げられます。
- 記述統計量の把握
	- 平均値・四分位数・標準偏差・最大値・最小値などの数値データ
	- 箱ひげ図による視覚的把握
	- サンプル図を挿入
- 単純なデータの可視化
	- 各説明変数（特徴量）と目的変数との関係性の可視化
		- 記述統計量で得られた値などを元にプロットします。
	- 散布図
		- 説明変数・目的変数の間を用いて点を２次元にプロットする。
	- 折れ線グラフ
	- ヒストグラム
		- データの分布を視覚的に把握します。
		- サンプル図を挿入
	- 相関係数
		- 変数間の相関性を数値的指標により求めることができます。
- 変換を加えたデータの可視化
	- 主成分分析（Principal Component Analysis: PCA）
		- 分散の大きさを指標としてデータの特徴をより強く表すように軸を取り直します。分散の大きい軸を2つか３つ取り出すことで可視化できます。



次に Kaggle Competition のデータを用いて上記の内容の一部を実際に求めてみたいと思います。



<!--
----------------------------------------
-->
## Kaggleのコンペティションデータを用いたEDA
コンペティションサイト : <a href="https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection">TalkingData AdTracking Fraud Detection Challenge</a>
今回取り扱う内容はオンライン広告のクリック詐欺に関するコンペです。
オンライン広告を通じてサイトを訪れた各クリックデータが以下のような特徴量を持っています。目的はオンライン広告がクリックされたときそれに付随する特徴量から、クリックした人が最終的にサイト内のアプリをダウンロードしたかどうかを予測するものです。特徴量の説明を軽く以下に示します。

- トレーニングデータ・テストデータともに含まれる変数
	- ip : IPアドレス
	- app : 広告が宣伝しているアプリ
	- device : モバイル機種
	- os : モバイルOS
	- channel : モバイル広告の出版社を示すID
	- click_time : 広告をクリックした日時
- トレーニングデータにのみ含まれる変数
	- distributed_time : もしユーザーがアプリをダウンロードした際はその日時（ダウンロードしていない場合は欠損）
	- is_distributed : アプリをダウンロードしたか否か（目的変数）


### メモリに乗り切らない問題
早速分析を始めるために学習データをKaggleのKernel上で読み出してみましょう。

```
import pandas as pd
df_train = pd.read_csv(“../input/train.csv”)
```

しばらく待っていると読み込みきる前に次のような画面が表示されます。

<img src='https://lh3.googleusercontent.com/kwnkUSmbgBjJfHNVCm-Ew7ujVMRwgCKeh6DLJXsmQvL50Fi18uFa7LXbBWNLzARJtFdFlTHX65WwJF4E3CPVQ8t4=s800'/>

どうやらカーネルが死んでしまったようです。これは与えられたデータセットのサイズが大きすぎて、メモリ(RAM)に乗り切らなかったためです。今回の学習データはおよそ２億件(7GB)のクリックデータから成り、通常の8、16GBのメモリを搭載したPCで読み込み処理するのは非常に困難なのです。
　今回のコンペに関してのみ言えば、小さめのデータセット(train_sample.csv)が事前に用意されていました。中身は学習データはランダムに10万件を取り出したCSVファイルです。確かに平均値などの値を求める分にはランダムな抽出でもさほど問題ないと思いますが、10万件だけを取り出したプロットは２億件全てを用いたプロットより視覚的に捉えられる特徴が減ってしまいます。

そこで、今回は２億件全てのデータをメモリの少ないPCでも扱える方法を紹介し、それを用いて分析しようと思います。


<!--
----------------------------------------
-->
## サイズの大きいデータの扱い方
手元に低スペックのPCしかない場合、以下のような手法を用いて巨大なデータを扱うことができます。
- CSVファイルから決まった行数だけ読み込んで処理を行う（標本抽出）
- 並列計算フレームワーク<a href="https://dask.pydata.org">DASK</a>を使い、全てのデータに対して処理を行う
- BigQueryなどのクラウドサービスを使用する。
    - クラウドサービスとして挙げられるGCPのGoogle BigQueryやAWSのAmazon Redshiftなどは巨大なデータを並列処理などを駆使して高速に計算してくれます。EDAや前処理、モデル構築などで頻繁に大量のデータにアクセスする機械学習分野でもよく用いられる手法のようです。

今回はDASKを用いた2つ目の手法を紹介します。


<!--
----------------------------------------
-->
## DASK
ではDASKについて説明する前に、早速先ほどのCSVファイルを読み込んでみましょう。
もしdaskをインストールしていない場合は`pip install dask`でインストールできます。
詳細は <a herf="https://dask.pydata.org/en/latest/install.html">Install Dask - DASK Docs</a>
```python
>>> import dask.dataframe as dd
>>> df_train = dd.read_csv(”../input/train.csv”)
```
これを実行すると１秒もかからずに終了します。この間に２億件ものデータを読み込んでいるとは考えられない速さです。（この速さについては後述します。）

さて、ではDaskとは何なのでしょうか。一言でいえば並列・分散処理のためのフレームワークといえます。。大きなデータを複数のブロックに分割し、各ブロックに分けることで処理を分散します。ブロック単位で分散することができるということは１度に全てのデータを読み込む必要はありませんから、メモリ消費のピーク値をかなり抑えることができます。
また、pandasなどでは大きいデータを扱えるようにchunksizeを利用した繰り返し処理を行うことができますが、Daskを利用することでよりメモリ消費量を抑えつつ計算速度も高めることができるようです。
（参照：<a href="https://qiita.com/kodai_sudo/items/c2ff1e85da18eaf13b65">データ分析のための並列処理ライブラリDask - Qiita</a>）


### DASKの基本的な処理
基本的な操作の１つとしてデータ全ての和・平均を求める計算をしてみましょう。

#### Irisデータの保存
簡単のためにscikit-learnのirisデータを使用したいと思います。CSVデータから読み込む例を扱いたいため一度、CSVデータとして保存します。
```python
>>> from sklearn import dataset
>>> iris = datasets.load_iris()
>>> columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
>>> df = pd.DataFrame(data=iris.data, columns=columns)
>>> df.to_csv(path_or_buf="iris.csv", index=False)
```

#### read_csv() - BLOCKSIZE
次にDASKで読み込んでみます。
```python
>>> df = dd.read_csv(urlpath="iris.csv")
>>> print(df.npartitions)
1
```

ここで`npartitions`というものが出てきましたが、これはCSVデータを何個のブロック単位に分けて処理をするかというものです。
このデフォルト値は実機のメモリ(RAM)の容量とCPUの個数に応じて決まります。ソースコード中での計算式は
`(実機のメモリサイズ) * (CPUの個数) // 10`
となっています。10で割った商としているのはある程度余裕をもたせる為だと思われます。
よって、小さいirisデータは分割されずに処理されてしまっているようです。

分割された処理を見てみるために`blocksize`を指定してみます。(blocksizeの単位はbyteです）
```python
>>> df = dd.read_csv(urlpath="iris.csv", blocksize=1500)
>>> print(df.npartitions)
2
```
２つに分割されましたのでこれを用いて和と平均を求めてみます。

#### 和・平均
基本的な操作はpandasと変わらないので`sum()`と`mean()`を使います。
```python
>>> df_sum = df.sum()
>>> print("sum¥n", df_sum)
sum
Dask Series Structure:
npartitions=1
petal length (cm)    float64
sepal width (cm)         ...
dtype: float64
Dask Name: dataframe-sum-agg, 9 tasks

>>> df_mean = df.mean()
>>> print("mean¥n", df_mean)
mean
Dask Series Structure:
npartitions=1
petal length (cm)    float64
sepal width (cm)         ...
dtype: float64
Dask Name: dataframe-mean, 13 tasks
```

この出力結果は期待したものと違いますね。
実はこれはDASKの`delayed()`という関数による結果です。次で詳しくみていきます。

#### delayed() と compute()
`delayed()`は関数を引数にとることで関数の実行そのものを任意の段階に遅らせることができます。つまり、計算の定義だけを先に組み上げて最後に一気に計算するといった感じです。
下の例で確かめます。

```python
>>> from dask import delayed
>>> def inc(x):
        return x + 1
>>> print("Not delayed: ", inc(10))
Not delayed:  11

>>> x = delayed(inc, pure=True)(10)
>>> print("delayed: ", x)
delayed:  Delayed('inc-6e45b15e167988c125e70c84590f449d')

>>> print("delayed compute: ", x.compute())
delayed compute:  11
```

`compute()`前の`delayed`の出力にはハッシュ値が添えられていますがこれは同じ関数を複数回呼びだされた時に区別するためのものなので気にする必要はありません。

また、`delayed()`で積み上げてある処理は`visualize()`によって可視化できます。
引数の`filename`はデフォルトで`mydask.png`になっていますが指定することができます。JupyterなどのIPython系であれば出力で画像が表示されます。
```python
>>> dinc = delayed(inc)
>>> layer = add(add(5, dinc(4)), add(3, add(1, 2)))
>>> print("output: ", layer.compute())
output:  16

>>> layer.visualize(filename="add_inc_vis.png")
```
<img src='https://lh3.googleusercontent.com/Mxlc-hqIOEv4hiyt1PiECZvjgxGDC4kuFXohBzdim3qW5Hn65LnoYLYsDA28-_T-mC_6oYYi_Lnf9wnVt5TsfT1r=s600'/>



ここまでで`delayed()`の仕組みと`compute()`の使い方が理解できたとおもいます。

では、和と平均の話にもどります。
`compute()`を使って実行しましょう。
```python
>>> df_sum = df.sum()
>>> print(df_sum.compute())
sepal length (cm)    876.5
sepal width (cm)     458.1
petal length (cm)    563.8
petal width (cm)     179.8
dtype: float64


>>> df_mean = df.mean()
>>> print(df_mean.compute())
sepal length (cm)    5.843333
sepal width (cm)     3.054000
petal length (cm)    3.758667
petal width (cm)     1.198667
dtype: float64
```
それらしい出力がかえってきました！

では`visualize()`で可視化してみます。
```python
>>> df_sum.visualize(filename="df_sum.png")
```
<!-- Insert Image URL -->
<img src='https://lh3.googleusercontent.com/Af9RsZSSBS96jlxYXgrPPh_h9uUkGtkfd9WpmRI9MseuJtjOploiUGy-jm15-fOAAB7ymZkp4pchwm4lrWT4_Q=s1700'/>

```
>>> df_mean.visualize(filename="df_mean.png")
```
<!-- Insert Image URL -->
<img src='https://lh3.googleusercontent.com/6gEVAcn8afI_96531GtS9IkQNrNcqykMgZO6HC6ccgk-Jszqp9fPY83i8RRQwhMvGpdAZA1fEeDTa4ARfW4d-qY=s1400'/>


これらの図の下の方を見てみると`read_block_from_file`と`pandas_read_text`という文字が丸で囲まれているのがわかります。
ソースコードをたどるとわかるのですが以下の一部にあるとおり、実はこれらは`read_csv()`の関数が呼び出している、`delayed`された関数なのです。

- <a href="https://github.com/dask/dask/blob/e1c48e0c970aeb81ffeb58791db2ca3ce76fa846/dask/bytes/core.py#L102">read_block_from_file</a>

```python
# `read_csv()`の一部

...
...
def read_block_from_file(lazy_file, off, bs, delimiter):
    with lazy_file as f:
        return read_block(f, off, bs, delimiter)
...
...
def read_bytes(urlpath, delimiter=None, not_zero=False, blocksize=2**27,
                sample=True, compression=None, **kwargs):
    ...
    ...
    delayed_read = delayed(read_block_from_file)                        # <---

    out = []
    for path, offset, length in zip(paths, offsets, lengths):
        token = tokenize(fs_token, delimiter, path, fs.ukey(path),
                         compression, offset)
        keys = ['read-block-%s-%s' % (o, token) for o in offset]
...
...
```

- <a href="https://github.com/dask/dask/blob/e1c48e0c970aeb81ffeb58791db2ca3ce76fa846/dask/dataframe/io/csv.py#L217">pandas_read_text</a>

```python
# `read_csv()`の一部
...
...
def pandas_read_text(reader, b, header, kwargs, dtypes=None, columns=None,
                     write_header=True, enforce=False):
    """ Convert a block of bytes to a Pandas DataFrame
    ...
    """

...
...
def text_blocks_to_pandas(reader, block_lists, header, head, kwargs,
                          collection=True, enforce=False,
                          specified_dtypes=None):
    """ Convert blocks of bytes to a dask.dataframe or other high-level object
    ...
    """
    ...
    columns = list(head.columns)
    delayed_pandas_read_text = delayed(pandas_read_text, pure=True)         # <---
    dfs = []
    for blocks in block_lists:
...
...
...
```


#### read_csv()の実行が速い理由
これでKaggleコンペの巨大なCSVデータに対して`read_csv()`が１秒もかからずに実行できた理由がわかりました。
つまり、`read_csv()`の処理は`delayed()`によって実際にはまだ実行されていない状態、即ちメモリ(RAM)に読み込んでいない状態となります。そして、`mean()`や`sum()`等の具体的数値を求める際に`compute()`をつけることで一連の処理の計算が始まります。


## EDAの実行
それでは`TalkingData`の特徴をつかむためにいくつかの値を計算・描画してみます。
（これ以降では`Jupyter Notebook`上で実行しています。スクリプトで実行する場合は`print`関数などで出力する必要がある場合があります。）
今回の一連の実行は以下で公開しています。
- <a href="https://github.com/pollenjp/talkingdata/blob/0a868207aab3e404de749932bdbc2604bd66dfa2/notebooks/eda_with_dask.ipynb" target="_blank">eda_with_dask.ipynb</a>
- <a href="https://github.com/pollenjp/talkingdata/blob/9d72165e73d57f871d56b58a119ec5315fb400b2/notebooks/eda_with_dask__time_pattern.ipynb" target="_blank">eda_with_dask__time_pattern.ipynb</a>

### データの読み込み
まずCSVデータを読み込んでいきます。
pythonでは型を指定する必要はありませんがあえて最小限のデータ型を調べて指定することでメモリを節約することができる場合があります。
<!--
modify
データ型を調べる過程も載せる
-->
```
import dask.dataframe as dd

dtypes = {
    "ip" : "uint32",
    "app" : "uint16",
    "device" : "uint16",
    "os" : "uint16",
    "channel" : "uint16",
    "is_attributed" : "uint8",
}

df_train = dd.read_csv(urlpath=str(train_csv_Path),
                       dtype=dtypes,
                       parse_dates=["click_time", "attributed_time"],
                      )
```
<img src='https://lh3.googleusercontent.com/TQ7-9e1MpMBXYCm52qThLRY7d7j3y4Mt6G5rbWco-xqrdilDYx3POB0h8QS7-yy5n83rcTIi98plzSB5cyvXJZKs=s600'/>


### 欠損値の確認
他の処理への影響がある欠損値をどれほど含んでいるかを調べてみます。
```
df_train.isnull().sum().compute()
```
実行時間 : 約4分
出力結果

<img src='https://lh3.googleusercontent.com/QxEVMRpG9NPeNYc0zRaoHwwiCxZbKnMLslP2WKNe_gK_ZXaWroFpOpx2U5x9u6KaHJC06Dq_GfrSawJoiH323WQ=s600'/>

`attributed_time`に欠損値が含まれているようですがこれは目的変数が1のときに与えられる特徴量であり、数えてみると目的変数が1の数と同じだけあります。よっておかしな欠損は無いことがわかります。
<!--
modify
-->


### 記述統計量の把握
記述統計量の計算はpandasの`describe()`で簡単に計算できてDASKでも実装されています。
```
df_train.describe().compute()
```
実行時間 : 約6分
出力結果
<img src='https://lh3.googleusercontent.com/YoHEz5YzZ4BGbRC_jNByY3FU4cc2oUJ95SdBdF_CLHn8HTAX8FZE_Unf3jx4MD7dNZXBqj5MRA30dV0U8k6Y0A=s600'/>


### 単純なデータの可視化
特にデータをいじることなく、から得られるデータを抽出して可視化してみます。

#### 目的変数の値の分布
`is_attributed`が0か1かを予測する上で全データに対するこれらがどの程度の割合で分布しているのかを知るのはモデル構築の際に重要です。
今回は0,1の2値なので`is_attributed`の平均値をとるとそれが`is_attributed`が1である割合であることがわかります。
```
mean = df_train["is_attributed"].mean().compute()
prop_df = pd.DataFrame(data=[[mean, 1-mean]],
                       index=None,
                       columns=["App Downloaded (1)", "Not Downloaded (2)"],
                     )

## Plot
title = "is_attributed_proportion"

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)

sns.barplot(data=prop_df,
            order=["App Downloaded (1)", "Not Downloaded (2)"],
            ax=ax,
           )
ax.set_title(title)
ax.set_xlabel(xlabel="is_attributed")
ax.set_ylabel(ylabel="Proportion")

for p, prop in zip(ax.patches, prop_df.loc[0]):
    height = p.get_height()
    ax.text(x=p.get_x() + p.get_width() / 2,
            y=height + 0.01,
            s="%f %%" % prop,
            ha="center",
           )
fig.savefig(fname=(title + ".png"))
```
出力結果
<img src='https://lh3.googleusercontent.com/yy8kwh49KT9wzeiAx_NFBkLe5S3aMAgjTPMVvPJmLTjiIw5Y3KOnKAbEJQ83gS2vZyC3PSDe-AtNzcsPR1RDtQ=s800'/>これを見るとわかる通りダウンロードされたクリック数の割合が圧倒的に少ないことがわかります。通常の分類モデルとして勾配降下法などで評価すると`Not Downloaded`に引っ張られて、AUROC(Area Under the Receiver Operating Characteristic curve)がうまく上がりません。そこで異常検知などのモデルも思いつくかと思います。
実際、`TalkingData`のコンペで優勝されたチームの一人である<a href="https://www.kaggle.com/flowlight">flowlight</a>さんは`Downloaded`と`Not Downloaded`の値が同じになるようにデータを削る手法ネガティブサンプリングを用いたようです。
こちらに Kaggle Tokyo の時の資料が上がっています。
- <a herf="https://www.slideshare.net/TakanoriHayashi3/talkingdata-adtracking-fraud-detection-challenge-1st-place-solution" target="_blank">TalkingData AdTracking Fraud Detection Challenge (1st place solution)</a>


#### 特徴量の固有な値の数
固有な値の数とは例えば`device`という特徴量で言えば全クリックにはいくつの種類のOSが使われたかということです。
もしある特定の`device`からダウンロード率(`is_attributed`率)が極端に低い場合はそのクリックの情報から`is_attributed`を予測するのは困難になり役に立たないと判断できそうです。
これはDASKの機能は使わず以下のように実装しましたが、forループを使ったためにかなり時間がかかりました。
```
features = ["ip", "app", "device", "os", "channel"]
uniques = [len(df_train[f].unique().compute()) for f in features]
uniques = pd.DataFrame(data=np.array(uniques).reshape(1, 5), columns=features)
unique
```
出力結果
<img src='https://lh3.googleusercontent.com/Z3y6ujqWjlG7rbOH9g6c4Qu2OVrWhcLFhaSND7-cMkwmss_R8GaZXyxQ4b9fO05S3JRjVoLjmJP28sQ-BFZTxYE=s600'/>


棒グラフで比較してみたいと思います。縦軸を対数関数にしている点に注意してください。
```
# 棒グラフでプロット
title = "uniques_counts_log10_plot"

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1, 1, 1)
val = np.log10(uniques.values[0])
print(val)

idx_cols = np.arange(len(uniques.columns))
b = ax1.bar(idx_cols, val)
ax1.set_xticks(idx_cols)
ax1.set_xticklabels(uniques.columns)
#ax1.set_ylim([0, 100])
ax1.set_ylabel('log10(y)')


color_list = ["#6666ff", "#ffb266", "#009900", "#000099", "#cc99ff"]
for i in idx_cols:
    b[i].set_facecolor(color_list[i])
for p, v in zip(ax1.patches, uniques.values[0]):
    ax1.text(x=p.get_x() + p.get_width() / 2,
             y=p.get_height() + 0.05,
             s=v,
             ha="center",
            )

ax1.set_title(title)
fig.savefig(fname=str(data_Path / "plot" / (title + ".png")))
```
出力結果
<img src='https://lh3.googleusercontent.com/ddAQsz2xcKVoOJrdgE_gckJ3phu9n7iiL1-nUENnHBTXoq101_ubaxzhfrunorQldSmv6MTBrlkNQpwwaf94YQ=s800'/>


#### クリック数の多さで並べ替え
クリック詐欺を考えたとき特定の種類の`ip`,`device`,`os`等から定期的にクリックするプログラムが動いていると予想しました。
ということはクリック詐欺が行われている特徴量は異常ににクリック数が多くなると考えられます。これらの関係もプロットして確認しようと思います。
各特徴量に対して行います。

##### IP
まずは各IPからクリックされた回数をカウントします。
扱いやすいようにDataFrameとして扱います。また、メモリ節約のため一時的に用いた変数も`del`を用いて削除します。
```
fea = "ip"
df_fea_val_counts_tmp = df_train[fea].value_counts()
# SeriesからDataFrameに変換
df_fea_val_counts_tmp = df_fea_val_counts_tmp.to_frame()
df_fea_val_counts = df_fea_val_counts_tmp.reset_index(drop=False)
del df_fea_val_counts_tmp
df_fea_val_counts.columns = [fea, 'counts']
df_fea_val_counts.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/0MQGTmDqA1f2wCBjFqPrE9NdWyjSGRLcGk-mZOzd57ycz-s4kIYRjo3JbOr68b3qn6Pb04P55gMjAlcdF6kJdg=s600'/>

`df_train`に`df_fea_val_counts`をマージします。SQLで言えば左結合と同等の処理になります。
```
df_train_tmp = dd.merge(left=df_train,
                        right=df_fea_val_counts,
                        how="left",
                        on=fea,
                       )
df_train_tmp.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/tN1EWPsm_qKtl-HrAtH32Mo1Vqk74o8vmAw959r4cyP3XxoW-XYP1K9gHZkuXwic7-ohG-doEiaYEaBOzjDXkzk=s600'/>


各IPのダウンロード率を求めます。
現段階では`groupby`や`reset_index`等の使用がpandasと異なっていたり、DASKでは実装されていないメソッドなどがあり、適宜ドキュメントで確認する必要があります。
```
proportion = df_train_tmp[[fea, 'is_attributed']].groupby(by=fea).mean().reset_index()
proportion.columns = [fea, "prop_downloaded"]
proportion.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/ubALwREsyJKPfbzJfUaVx8gkPSwoEePDu5lfeVCGZhpY4Ed68Kbbxm2nTz72LXE165j5CMoDbcyvMZxIPu6giss=s600'/>


各IPのクリック数を求めて降順にソートします。
```
n = 1000000  # 1000000 > 277396
counts = df_train_tmp[[fea, 'is_attributed']].groupby(by=fea).count().reset_index().nlargest(n=n, columns='is_attributed').reset_index(drop=True)
counts.columns = [fea, "click_count"]
counts.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/7BqzEYBMWm5XEJtGbQ7tEBqS3zFbDdR-DQznVLMrc2ERLnrIxiQYMZVzoPclj0P24WSGoRyddj5Dug6TczZJcdU=s600'/>


```
df_merge = counts.merge(right=proportion, on=fea, how='left', )
df_merge.columns = [fea, 'click_count', 'prop_downloaded']
df_merge.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/9ut38UJ2tZaThZmMM-YId8h8aomurG2kmBHdJWRT6D6HkaaCFqrYy_f5F_xolO8YsIm-o-3PM4BCcnAxTQxGWNo=s600'/>


クリック数上位300ほどをプロットします。縦軸を左と右とで分けて表示しています。`ax.twinx()`を用いると異なるグラフを重ねることができます。
```
n_plot = 300
title = "Conversion_Rates_over_Counts_of_300_Most_Popular_"+fea.upper()+"s"

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)


ax1.set_yscale("linear")
ax1.yaxis.tick_left()
df_merge[['click_count']][:n_plot].compute().plot(ax=ax1, color="blue",)
ax1.yaxis.set_label_position("left")
ax1.set_ylabel(ylabel='Count of clicks')
ax1.legend(loc=3, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ax2 = ax1.twinx()
ax2.yaxis.tick_right()
df_merge[['prop_downloaded']][:n_plot].compute().plot(ax=ax2, color="green",)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel(ylabel='Proportion Downloaded')
ax2.legend(loc=3, bbox_to_anchor=(1.05, 1 + 0.07), borderaxespad=0.)

ax2.set_title(title)
ax2.set_xlabel("Sort by " + fea)
fig.savefig(fname=title)
```
出力結果
<img src='https://lh3.googleusercontent.com/oTuf-1Iw_sX610f2sEaulTer52jTfHqngH8Wo3Zjswn3PmwFL6Ib7QnIyRTgvxxaOu0f-6EEEMlJkaC6ODP29w=s850'/>

この結果をからは上位300の`id_attributed`率はあまり変わらず0.02程度であることがわかります。これほど低いとこの情報から`is_attributed`の値を予測するのは厳しそうです。


次は逆に下位300をプロットします。
```
n_plot = 300
title = "Conversion_Rates_over_Counts_of_300_Worst_Popular_"+fea.upper()+"s"

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_yscale("linear")
ax1.yaxis.tick_left()
df_merge[['click_count']].compute()[-n_plot:].plot(ax=ax1, color="blue", lw=2.0)
ax1.yaxis.set_label_position("left")
ax1.set_ylabel(ylabel='Count of clicks')
ax1.legend(loc=3, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
#ax1.set_ylim(bottom=-0.01, top=1.05*df_merge[['click_count']][-n:].max(axis=0)['click_count'])

ax2 = ax1.twinx()
ax2.yaxis.tick_right()
df_merge[['prop_downloaded']].compute()[-n_plot:].plot(ax=ax2, color="green", lw=2.0)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel(ylabel='Proportion Downloaded')
ax2.legend(loc=3, bbox_to_anchor=(1.05, 1 + 0.07), borderaxespad=0.)

ax2.set_title(title)
ax2.set_xlabel("Sort by " + fea)
ax2.set_ylim(bottom=-0.01, top=1.05*df_merge[['prop_downloaded']].compute()[-n_plot:].max(axis=0)['prop_downloaded'])

fig.savefig(fname=str(data_Path / "plot" / title))
```
出力結果
<img src='https://lh3.googleusercontent.com/lr_rBYvX5Fk10nHHDdFdkBukWLGkbUl65CG77J7F-B0cGo4wi-LCn3uE0kshqAOlnQDm1wCFeySuX5aidfR_vIU=s850'/>

下位300に関しては逆に`id_attributed`率が高く、クリック数が少ない（1回のみ）ようです。これだけから結論付けるのは早計ですがクリック数が極端に少ない情報から`id_attributed`が予測しやすそうな印象を受けます。

他の特徴量については結果だけ示します。詳しくは<a href="https://github.com/pollenjp/talkingdata/blob/0a868207aab3e404de749932bdbc2604bd66dfa2/notebooks/eda_with_dask.ipynb" target="_blank">eda_with_dask.ipynb</a>をご覧ください。


#### APP
<img src='https://lh3.googleusercontent.com/D-6ePgrOuOROQnnliQxc067w_qSOVpl86sa64m1K_0h3yBOnBRBXECiYKsmdtA52GikuzUrBg9f3cg9McSzGvk8=s850'/>

<img src='https://lh3.googleusercontent.com/rkyMShfMjjSgGXW85sFFwlfbW-Uk_GeKw8Lm8f9eLmimhJZODcCIZL0jtjKkM6M8CuuQlWIfBLQJENpmWrPslE8=s850'/>


`App`に関してはクリック数が群を抜いて大きい箇所（上位２０位以内ほど）ではダウンロード率が低く、３０位以降からはノイズのようにダウンロード率がところどころ跳ね上がっています。このことから`App`に関してもクリック数の多さが関係していると考えられます。

##### Device
<img src='https://lh3.googleusercontent.com/wc1bV_qfSjAJqMb3Av4sPcixIjXgVKu9-2gxi5dXv0lRlHtiYZ9NB-MTqPzjp6PQ1NgcLfBp99WeUfqMFYakIw=s850'/>

<img src='https://lh3.googleusercontent.com/CZRd1lEXSmx006ki__Vpy9BZoQrLEyTeQ6yXm99Zqu5utWdzVDea6LZIKnM8-47NjhxgXgy6aYve7UyT4NPmGjE=s850'/>

`Device`のグラフではクリックの関係がわかりやすくあらわれています。上位5位以内ほどで極端にクリック数が多く、それ以降のDeviceでは平均して20%ものダンロード率を持っています。
先ほどの結果も合わせて`App`や`Device`という特徴量が予測に大きく寄与する可能性を感じられます。


##### OS
<img src='https://lh3.googleusercontent.com/q5Bq6qn5QZuMnVMrl0tYoLf-ixZLz9YSYytwzQB99gKxJUbiwq4Hxl1GNJOxgokkJa0-cWoBETQfm-8wJiGZoGuc=s850'/>

<img src='https://lh3.googleusercontent.com/4fHtSEkCwCgFAi6DTTBXwzdH9Q4vPLlHJppKvaJLeHWrbntZYLBAR1QiTneH76x3uBmUt8FmH_iyeWgGRbyTPw=s850'/>

`OS`に関してはずば抜けて高いダウンロード率を誇るOSがあるようです。しかし、他のOSには目立ったダウンロード率が見えにくいプロットです。収穫としてはこのずば抜けたOSを意識にとどめておくと後々他との関連が見えてきそうです。

##### Channel
<img src='https://lh3.googleusercontent.com/jQS7ZVVZwJJjPo1aHL-Zpsagv4rF_GS-IE5KitpehpiHTmtiGKziCc-65BaRui8slxRg8vud_nm-GmatcieNcyw=s850'/>

<img src='https://lh3.googleusercontent.com/-6N494po-d3kKkZqh6Nl4k0ZDxW5XZ5IV18GjZzCLSXWXcn6HaWOVZmjEMTVOzmjA9ebX7nW4bV6C-CeuCSCbQ=s850'/>

`Channel`においては他と少し違い、クリック数の順位が120位〜180位のダウンロード率が高めのようです。一概にクリック数に反比例するパターンではない特徴量であるという点が把握できます。

#### 時系列順に並べ替え
##### 日時順にダウンロード率の表示
データにクリック日時が含まれているのでこれを元に日時に沿ったクリック数とダウンロード率をプロットしてみます。
まずクリック時刻をまとめるために時間(hour)で丸めます。
```
section = "datetime"
fea = "click_rnd"
df_train[fea] = df_train["click_time"].dt.round('H')
click_counts = df_train[[fea,"is_attributed"]].groupby(by=fea).count().reset_index()
click_counts.columns = [fea, "click_counts"]
click_counts.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/rLf6CyvdULQgnYccVqyiXdFlJMor6Wtm6yEtq5ltiBVmIm6pCpr6_Ef7U7rdTlL9-ERUSeu9Taz-lOvCoLvJ00w=s850'/>


次にダウンロード率を求めます。
```
prop_downloaded = df_train[[fea,"is_attributed"]].groupby(by=fea).mean().reset_index()
prop_downloaded.columns = [fea, "prop_downloaded"]
prop_downloaded.head()
```
出力結果
<img src='https://lh3.googleusercontent.com/9PAtgPaQsTCkzE0LkHliZi3ROmh3dOgCLLuAFUV7kKtvPtIoz0_QAV4oK2kNg3q5XgQ7q_lQt0oSnBSbvDF3UQ=s850'/>


クリック数とダウンロード率をプロットします。
```
# plot
title = section + "_" + fea + "_click_counts_prop_downloaded"

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)


ax1.yaxis.tick_left()
ax1.yaxis.set_label_position("left")
click_counts[["click_counts"]].compute().plot(ax=ax1, color="blue")
ax1.set_ylabel(ylabel="click_counts")
ax1.legend(loc=3, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ax2 = ax1.twinx()
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
prop_downloaded[["prop_downloaded"]].compute().plot(ax=ax2, color="green")
ax2.set_ylabel(ylabel="prop_downloaded")

ax2.set_title(title)
ax2.set_xlabel(fea)
ax2.legend(loc=3, bbox_to_anchor=(1.05, 1 - 0.07), borderaxespad=0.)

ax1.set_title(label=title)
fig.savefig(fname=title)
```
出力結果
<img src='https://lh3.googleusercontent.com/RVSTmSCJHjsYwKmZ_TZzZ_1DEjeTW0mvs1NOr36Vttbjmu-KuDnnD4ldzuVD-e5IKB-NplKnJDp2vMopBFjQa8M_=s850'/>

この結果から一日単位で周期的な特徴が伺えます。そこで日付を無視した時刻のみのグラフを作成します。


##### 時刻順にダンロード率を表示
時間単位(hour)の各時刻で丸めてクリック数とダウンロード率を表示します。（コードは公開したコードを参照）
なおグラフの左上の`le7`は桁が`10^7`であることを示します。
<img src='https://lh3.googleusercontent.com/T1C5jTTwa3pzRz-bgjVzhrRtvMOgW3Sf1Mf8TUlC0t83iwdR8rHMbZpflCy-m-N_zrYRzy_lfSWJ-uJU8S5gJw0=s850'/>
図より15時から24時あたりのクリック数が下がっているのに対してダウンロード率が少し盛り上がる箇所があります。ダウンロード率が低いことから直接的には予測できないかもしれませんが、データの特徴が現れた箇所と言えます。

### 変換を加えたデータの可視化のために
最初のほうで主成分分析などによる可視化の手法を書きましたが、主成分分析をする際には機械学習ライブラリが必要となります。特に有名なものとして`sckikit-learn`というライブラリが広く使われていますが、今回ほどの大きなデータセットに対してはそのままの形で扱うことはできません。しかし、`DASK`がpandasの他に提供しているscikit-learnのスケーラブルなAPIを活用すれば可能になります。
- <a href="https://dask-ml.readthedocs.io/en/latest/index.html" target="_blank">Dask-ML</a>

これを用いることで巨大なデータセットに対しても機械学習を行うことができます。
今回は詳しく扱いませんが`DASK`が提供するスケーラブルなフレームワークを駆使することで、データをうまく変換し、生のデータからだけでは得られない特徴を把握することが可能になるかもしれません。


## 最後に
この記事では前半で探索的データ分析(EDA)の概要とDASKという並列計算フレームワークについての説明をし、後半でKaggleコンペのデータセット`TalkingData`を用いたEDAの一例を紹介しました。データが巨大な分、実行時間はかなり必要となりますが、今まで低スペックなPC故に読み込むことすらできなかった状態からデータの特徴を把握できるほどの手段は得られたのでは無いでしょうか。


### 本記事のコード
- <a href="https://github.com/pollenjp/talkingdata/blob/975f4023a4e419c7e7946b2bcb12486b1ce332f8/notebooks/dask_example.ipynb" target="_blank">dask_example.ipynb</a>
- <a href="https://github.com/pollenjp/talkingdata/blob/0a868207aab3e404de749932bdbc2604bd66dfa2/notebooks/eda_with_dask.ipynb" target="_blank">eda_with_dask.ipynb</a>
- <a href="https://github.com/pollenjp/talkingdata/blob/9d72165e73d57f871d56b58a119ec5315fb400b2/notebooks/eda_with_dask__time_pattern.ipynb" target="_blank">eda_with_dask__time_pattern.ipynb</a>

### 参考文献
- <a href="http://datanerd.hateblo.jp/entry/2017/09/09/200412" target="_blank">モデリング前の探索的データ分析(EDA)の典型例</a>
- <a href="https://www.msi.co.jp/splus/products/eda.html" target="_blank">S-PLUS: EDA</a>
- <a href="https://qiita.com/kodai_sudo/items/c2ff1e85da18eaf13b65" target="_blank">データ分析のための並列処理ライブラリDask - Qiita</a>
- <a href="https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns" target="_blank">TalkingData EDA plus time patterns - yulia</a>
