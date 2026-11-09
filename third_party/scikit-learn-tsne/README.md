# scikit-learn t-SNE 源码快照

本目录保留 scikit-learn 官方 t-SNE 的部分源代码，用于阅读与溯源。

- 上游项目：https://github.com/scikit-learn/scikit-learn
- 上游版本：1.9.1
- 上游提交：866c0f51e7560ef0303cbcc5f159df5382ea9e3f
- 许可证：BSD 3-Clause，完整原文见本目录的 `LICENSE`
- 官方 API 文档：https://scikit-learn.org/1.9/modules/generated/sklearn.manifold.TSNE.html

`sklearn/manifold/` 下的三个文件原样来自上游对应路径：`_t_sne.py`、`_barnes_hut_tsne.pyx` 和 `_utils.pyx`。它们依赖 scikit-learn 的其他模块、NumPy、SciPy 和已编译的 Cython 扩展；本目录不是完整安装包，不能把这些文件单独复制后运行。

如需使用 t-SNE，建议安装完整的 scikit-learn，再调用官方 API。版本 1.9.1 要求 Python 3.11 或更新版本；依赖版本按官方安装要求配置。示例只做特征降维，特征必须先由外部模型提取：

```bash
python -m pip install 'scikit-learn==1.9.1'
```

```python
import numpy as np
from sklearn.manifold import TSNE

# features.npy 的每一行对应一张图片的特征向量；总行数必须大于 30。
features = np.load("features.npy", allow_pickle=False)
xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
np.save("tsne_coordinates.npy", xy)
```

上述参数仅为示例，并非 PASB 论文图 6 的已证实设置。若比较多个模型，使用同一固定特征提取器处理所有图像，并共同拟合 t-SNE；二维坐标不能单独证明细胞表达正确。
