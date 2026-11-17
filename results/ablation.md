| Factor | Change | Params | MMACs | Val MAE | ΔMAE (in seed SDs) | Train Wh | ΔWh |
|---|---|---:|---:|---:|---:|---:|---:|
| Reference (3 seeds) | own_e15 config | 97,457 | 65.5 | 1.287 ± 0.067 | – | 0.227 | – |
| Pooling | sum -> global average + linear | 97,457 | 65.5 | 1.203 | -0.084 (-1.3) | 0.219 | -4 % |
| Receptive field | drop c5: RF 46 -> 30 px | 60,465 | 58.4 | 1.257 | -0.030 (-0.4) | 0.201 | -11 % |
| Input resolution | 96x128 -> 48x64 | 97,457 | 16.4 | 1.414 | +0.127 (+1.9) | 0.053 | -77 % |
| Width | 16-32-64 -> 8-16-32 channels | 24,537 | 16.8 | 1.238 | -0.049 (-0.7) | 0.091 | -60 % |
| Precision | fp32 -> fp16 autocast (training) | 97,457 | 65.5 | 1.237 | -0.051 (-0.8) | 0.198 | -13 % |
