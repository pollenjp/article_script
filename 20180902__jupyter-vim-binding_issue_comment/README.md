
date : 2018 / 09 / 02

You can't Copy and Paste cells by Ctrl-c, Ctrl-v with `jupyter-vim-binding`
However, you can do it like this below.

0. Move to Jupyter's HOME browser tab and disable `vim-binding` nbextension.
0. Move to a notebook that have some cells you want to copy, and save a notebook by `Ctrl-s` in no cell insert mode.
0. Then reload the notebook, select cells and `Ctrl-c` to copy them.
0. Move to a notebook you want to paste them, and also save a notebook by `Ctrl-s` in no cell insert mode.
0. Select a cell you want to paste copied ones above and then `Ctrl-v` paste cells.
0. Finally, move to Jupyter's HOME browser tab and enable `vim-binding` nbextension. 


[jupyter-vim-binding copy-paste ctrl-c ctrl-v - YouTube](https://youtu.be/T8dyiFILW4I)

![Alt Text](./jupyter-vim-binding_copy-paste_ctrl-c_ctrl-v.gif)


[Copy-Paste cells between notebooks doesn&#39;t work in Firefox · Issue #126 · lambdalisue/jupyter-vim-binding · GitHub](https://github.com/lambdalisue/jupyter-vim-binding/issues/126#issuecomment-417932921)

