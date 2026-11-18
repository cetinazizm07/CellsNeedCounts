| Model | BBBC039/naive | BBBC039/matched | BBBC039/tiles | BBBC002/naive | BBBC002/matched | BBBC001/naive | BBBC001/matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| *mean true count* | *118* | *118* | *118* | *69* | *69* | *360* | *360* |
| Own CNN final (48x64, sum) | 0.677 | **0.183** | **0.165** | 0.704 | **0.597** | 0.916 | **0.034** |
| Own CNN reference (96x128, sum) | 0.680 | 0.214 | 0.194 | 0.717 | 0.689 | 0.898 | 0.211 |
| Own CNN reference, avg pool | 0.681 | 0.804 | 0.208 | 0.705 | 0.804 | **0.877** | 0.956 |
| MobileNetV3-S FT | 0.715 | 0.829 | 0.307 | 0.753 | 0.832 | 0.930 | 0.953 |
| ResNet-18 FT | **0.645** | 0.716 | 0.192 | **0.668** | 0.708 | 0.897 | 0.885 |
| MobileNetV3-S scratch | 0.840 | 0.984 | 0.421 | 0.832 | 0.944 | 0.959 | 0.984 |
| ResNet-18 frozen | 0.650 | 0.714 | 0.170 | 0.755 | 0.726 | 0.928 | 0.900 |

Signed bias (cells, + = over-count):

| Model | BBBC039/naive | BBBC039/matched | BBBC039/tiles | BBBC002/naive | BBBC002/matched | BBBC001/naive | BBBC001/matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| Own CNN final (48x64, sum) | -75.4 | -4.4 | +0.1 | -48.5 | +41.2 | -330.0 | +3.6 |
| Own CNN reference (96x128, sum) | -75.5 | -8.1 | -4.1 | -49.4 | +47.5 | -323.5 | +75.9 |
| Own CNN reference, avg pool | -75.4 | -90.3 | -4.1 | -48.6 | -55.4 | -316.0 | -344.3 |
| MobileNetV3-S FT | -80.7 | -94.7 | -22.9 | -51.9 | -57.3 | -334.9 | -343.4 |
| ResNet-18 FT | -72.1 | -80.4 | -7.8 | -46.0 | -32.5 | -323.0 | -318.8 |
| MobileNetV3-S scratch | -94.6 | -110.9 | -33.5 | -57.3 | -65.1 | -345.3 | -354.6 |
| ResNet-18 frozen | -75.9 | -83.8 | -16.5 | -49.5 | -49.7 | -334.2 | -324.1 |
