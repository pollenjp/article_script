"""Utility module."""
#  reference : 
#    - https://github.com/JihongJu/deep-pu-learning
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#####################################################################################################
#  LOG PROCESS BAR
#####################################################################################################
def log_progress(sequence, every=None, size=None, name='Items', enum=False, bar=True):
    """
    cite : alexanderkuk/log-progress
      - Jupyter Notebook
      - https://github.com/alexanderkuk/log-progress
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    if not bar:
        for idx, record in enumerate(sequence, start=1):
            if not enum:
                yield record
            else:
                yield idx,record
        return

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            if not enum:
                yield record
            else:
                yield index, record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
    return


#####################################################################################################
def fit_and_plot(X, Y, classifier=None, marker_size=None, Y_true=None):
    """
    Fit and plot.
    Purpose
        - Classification
    X : np.array, shape: (batch, features)
    Y : np.array, shape: (batch, one-hot-encoded_classes)
    """
    # from IPython import get_ipython
    # get_ipython().run_line_magic('matplotlib', 'inline')

    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    min_size, max_size = 5, 30  # min and max mark size
    default_size = 10
    if marker_size is None:
        marker_size = ["default"]
    if not isinstance(marker_size, list):
        marker_size = [marker_size]

    n_samples, n_classes = Y.shape
    n_rows = len(marker_size)
    n_cols = n_classes - 1
    S = np.ones((X.shape[0], n_rows)) * default_size  # default maker size

    f, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if isinstance(axs, matplotlib.axes.SubplotBase):
        axs = np.array([axs])
    if n_rows == 1:
        axs = np.expand_dims(axs, 0)
    elif n_cols == 1:
        axs = np.expand_dims(axs, 1)

    if classifier:
        # we create an instance of Neighbe the number of bootstrap resamples
        # (n_boot) or set ci to None.ours Classifier and fit the data.
        classifier.fit(X, Y)
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        def normalize_size(s, y_enc, min_size=5, max_size=30):
            """Normalize S to (min_size, max_size)."""
            N, K = y_enc.shape
            for k in range(K):
                is_k = y_enc[:, k] > 0.
                s_k = s[is_k]
                s_k = (s_k - np.min(s_k)) / (np.max(s_k) - np.min(s_k))
                s[is_k] = s_k * (max_size - min_size) + min_size
            return s

        for j in range(n_rows):
            cmaps = [plt.cm.RdBu, plt.cm.RdYlGn]
            for k in range(1, n_classes):
                z = Z[:, k]
                # Put the result into a color plot
                zz = z.reshape(xx.shape)
                axs[j, k - 1].contourf(xx, yy, zz,
                                       cmap=cmaps[k - 1], alpha=.6)
                # axs[j, k - 1].set_title(
                #     'Class {} probabilty, per instance {}'.format(
                #         k, marker_size[j]))

            if marker_size[j] == "gradient":
                G = classifier.calc_gradient(X, Y)
                G = np.abs(G)
                y = np.argmax(Y, axis=1)
                s = G[np.arange(n_samples), y]
                s = normalize_size(s, Y, min_size, max_size)
            elif marker_size[j] == "loss":
                L = classifier.calc_loss(X, Y)
                s = L.copy()
                s = normalize_size(s, Y, min_size, max_size)
            S[:, j] = s.copy()

    # Plot also the training points
    ms = ['o', 's', '^']
    cs = ['r', 'b', 'g']

    n_samples = Y.shape[0]
    y = np.argmax(Y, axis=1)
    if Y_true is None:
        y_flip = np.zeros(n_samples, dtype=bool)
    else:
        y_true = np.argmax(Y_true, axis=1)
        y_flip = np.logical_and(y == 0, y_true != 0)

    for k in range(1, n_classes):
        for j in range(n_rows):
            for idx in range(n_samples):
                if y_flip[idx] > 0:
                    axs[j, k - 1].scatter(X[idx, 0], X[idx, 1], c=cs[y[idx]],
                                          marker=ms[y[idx]], s=10 * S[idx, j],
                                          edgecolor='black', linewidth='2',
                                          linestyle='dotted')
                else:
                    axs[j, k - 1].scatter(X[idx, 0], X[idx, 1], c=cs[y[idx]],
                                          marker=ms[y[idx]], s=10 * S[idx, j],
                                          edgecolor='black', linewidth='2',
                                          linestyle='solid')
            # axs[k - 1, j].set_xlabel('$x_0$')
            # axs[k - 1, j].set_ylabel('$x_1$')

            axs[j, k - 1].set_xlim(xx.min(), xx.max())
            axs[j, k - 1].set_ylim(yy.min(), yy.max())
            axs[j, k - 1].set_xticks(())
            axs[j, k - 1].set_yticks(())
    # plt.show()
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    return plt


def synthesize_pu_labels(Y, pct_missings=None, random_state=None,
                         verbose=False):
    """Synthesize PU labels."""
    # To PU
    if pct_missings is None:
        pct_missings = np.arange(0., 1 + 1e-8, 0.1)
    Y_pu = {}
    np.random.seed(random_state)
    n_samples = len(Y)
    for pct in pct_missings:
        y = np.argmax(Y, axis=1)
        flip = np.random.rand(n_samples)
        # Positive の中で <flip> が <pct> より小さくなったときの index 要素を 0 (unlabel) に変換
        y[(y != 0) & (flip < pct)] = 0
        Y_pu[pct] = np.eye(Y.shape[1])[y]
        if verbose is True:
            print('Positive (pct_missing={:2.1}):'.format(pct),
                  np.sum(np.argmax(Y, 1)>0), ' -> ', np.sum(y>0))
    return Y_pu


#####################################################################################################
def synthesize_pn_labels(Y, pct_missings=None, random_state=None,
                         verbose=False):
    """Synthesize PN labels."""
    # To PN
    if pct_missings is None:
        pct_missings = np.arange(0., 1 + 1e-8, 0.1)
    idx = {}
    np.random.seed(random_state)
    n_samples = len(Y)
    for pct in pct_missings:
        y = np.argmax(Y, 1)
        flip = np.random.rand(n_samples)
        mask = flip < pct
        y[mask] = -1
        idx[pct] = np.where(np.invert(mask))[0]
        if verbose is True:
            print('Positive vs. Negative (pct={}):'.format(pct),
                  np.sum(y > 0), ' vs.', np.sum(y == 0))
    return idx
