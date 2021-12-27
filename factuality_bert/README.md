# factuality_bert

This repository contains code and data release for the following paper:

[Nanjiang Jiang, Marie-Catherine de Marneffe. He knows better than the doctors: BERT for event factuality fails on pragmatics. TACL 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00414/107616/He-Thinks-He-Knows-Better-than-the-Doctors-BERT)

### Getting started

First, follow the [tutorial](tutorials/setup_tutorial.md) to install dependencies and set up environments.


The directory [factuality_scripts](factuality_scripts) contains all the scripts to train the models experiments in the paper. They can be run as follows:
```
./factuality_scripts/factuality_single_task.sh CB 42
```

### Data
The directory [glue_data](glue_data) contains all the preprocessed data and splits used in the paper.

 
### License

This package is based on [jiant v1](https://github.com/nyu-mll/jiant-v1-legacy).
`jiant` is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.


