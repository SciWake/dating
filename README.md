# dating

### mn.tflite - архитектура
```
INPUT
[1, 224, 224, 3] uint8
         │
         ▼
CAST → float32
         │
         ▼
Conv 1×1 (3 → 3)
Поканальная нормализация
         │
         ▼
CONVOLUTIONAL BACKBONE
         │
         ├── 112×112×32
         ├──  56×56×32
         ├──  28×28×64
         ├──  28×28×192
         ├──  14×14×96
         │      └── 5 residual blocks
         ├──   7×7×128
         │      └── 5 residual blocks
         └──   7×7×960
                    │
                    ▼
           Global Average Pooling
                    │
                    ▼
                g [960]
                    │
       ┌────────────┼────────────────────┐
       │            │                    │
       ▼            ▼                    ▼
  FC 960→512   FC 960→128          CONCAT(g,e)
    Linear        ReLU                [1472]
       │            │                    │
       ▼            ▼                    ▼
    e [512]       b [128]         FC 1472→128
       │                                ReLU
       ├──────────────┐                  │
       │              │                  ▼
       ▼              ▼                c [128]
 OUTPUT 0       FC 512→512               │
 Embedding         ReLU                  │
                      │                  │
                      ▼                  │
                    a [512]              │
                      │                  │
                      └──────┬───────────┘
                             │
                   CONCAT(b, c, a)
                             │
                             ▼
                          h [768]
                             │
                             ▼
                         FC 768→6
                             │
                             ▼
                           SIGMOID
                             │
                             ▼
                           OUTPUT 1
                            [1, 6]
```
