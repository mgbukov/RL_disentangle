## Configs

For training agents on small system sizes (< 10 qubits) use `train.py`.
`train.py` uses the NumPy implementation of the RL environment and it's faster.

For training large system sizes (>= 10 qubits) use `trainL.py` and `env_device="cuda"`.
`trainL.py` uses PyTorch implementation of the RL anvironment and computation
can take place on a GPU.

---

The table below shows which train script to use for each config:

| config                    | script        |
| ---                       | ---           |
| 4q-final.yaml             | train.py      |
| 5q-final.yaml             | train.py      |
| 6q-final.yaml             | train.py      |
| 12q-eta4.1-stage1.yaml    | trainL.py     |
| 12q-eta4.1-stage2.yaml    | trainL.py     |
| 16q-eta4.1-stage1.yaml    | trainL.py     |
| 16q-eta4.1-stage2.yaml    | trainL.py     |