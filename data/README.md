# Datasets Dir

Make sure that datasets are stored as follows:

```
├── data/
│   ├── FHM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   │   └── train_with_explanations.jsonl
│   ├── HarM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   │   └── train_with_explanations.jsonl
│   └── MAMI/
│       ├── images/
│       │   └── ...
│       ├── test.jsonl
│       └── train.jsonl
│   │   └── train_with_explanations.jsonl
└── ...
```

**NOTE:** The `images/` folder is not provided in this repository due to storage limitations. Please download the image files from the official releases of the FHM, HarM, and MAMI datasets and place them in the corresponding `images/` directory before running the code.