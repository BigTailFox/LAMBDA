# LAMBDA

**L**atent-**A**ction **M**ulti-**B**eam Search with **D**ensity-**A**daption

`LAMBDA` is a SMBO(Sequential Model-based Black-Box Optimization) Method derived from [LA-MCTS](https://github.com/facebookresearch/LaMCTS) for both black-box optimization and coverage problem. This repo contains the official implementation of `python` in this [pre-print](https://arxiv.org/abs/2203.13708), as well as some artificial functions and test problems to evaluate the algorithm. Feel free to reproduce our results on test functions in 5 minutes.

**NOTICE** This repo is still in development, and the `dev` branch could have some unknown bugs. We would appreciate it much if you find one and issue us.

## Black-Box Coverage Problem

`LAMBDA` is designed mainly for the Black-Box Coverage (BBC) problem, which means, to estimate the level-set of the inequality $f(x)<y^*$ with a limited budget to evaluate the black-box function $f(x)$, which is usually expensive. We first formalize the BBC problem from the safety evaluation of the automonous driving system in a logical scenario space, and also believe that the problem abstaction of BBC and the LAMBDA algorithm can be scaled to related scenarios in other fields with little change.

Here is the benchmark results of ours method with a bunch of classical or SOTA methods such as TuRBO, BO, GA, etc.

![benchmark1](doc\readme\picture1.png)
![benchmark1](doc\readme\picture2.png)

Refer to the [pre-print](https://arxiv.org/abs/2203.13708) for a detailed introduction of our work. There are also some further works based on `LAMBDA` coming soon.

## Dependencies

Need `python>=3.7`, and packages in `requirements.txt`, using venv or conda is recommanded.

## Usage

See `examples/hoelder_table_lambda.py` for detail.

## License

This repo can be distibuted under the MIT license.

