# Guiding exploration via invariant representations

The long accepted norm of the modern machine learning systems is that they are good at one thing, namely the one they have been initially trained on. Some of the more recent developments have expanded the capabilities of a single agent to multitude of tasks, but a more general approach of leveraging past knowledge is still in its infancy. The sub-field of machine learning known as *lifelong learning* aims to build and analyze systems that continuously learn by accumulating past knowledge, that is then used in future learning and problem solving. The aim of this work is to explore a particular application of lifelong learning paradigm to reinforcement learning. More specifically, my aim is to test a new policy, that I called *Lifelong-Learning Deep Q-Network (LLDQN)*, and evaluate agent's behavior, stability of learning and obtained reward compared to a regular *DQN* policy.

The LLDQN method is described in great detail in the [attached PDF report](./src/data/assets/work.pdf). The report was limited to $3$ full pages, not including references. Due to their small size, I have attached trained models in the `src/data/models` directory, including action and observation autoencoders as well as baseline and LLDQN policies. This project was done as part of the Optimization for Machine Learning course held by [Sebastian U. Stich](https://www.sstich.ch) ([CISPA Helmholtz Center for Information Security](https://cispa.de), [Saarland Informatics Campus](https://saarland-informatics-campus.de)).

## Runtime

The reported results are generated by scripts found in the `scripts` module. If not otherwise specified, we assume the commands are run from the repository root.

### Training

I now briefly describe the training procedure for different components of a system.

#### Autoencoders

To train both, action and observation autoencoders, you can run the following:

```
python src/scripts/train_autoencoder.py
```

Feel free to change the tasks (i.e., environments) for which the training procedure is run.

#### Baseline Policy

To train the baseline (i.e., reference) DQN policy, you can run the following:

```
python src/scripts/train_baseline.py
```

Feel free to change the tasks (i.e., environments) for which the training procedure is run.

#### LLDQN Policy

To train the LLDQN policy, you can run the following:

```
python src/scripts/train_lldqn.py
```

Feel free to change the tasks (i.e., environments) for which the training procedure is run. As a initial starting point, I have provided the training procedure for `Acrobot-v1` environment, leveraging baseline policy learned for `CartPole-v1` environment, as well as a training procedure for `CartPole-v1` environment, leveraging baseline policy learned for `Acrobot-v1` environment.


### Evaluation

I now briefly describe the procedure used to generate reported plots and figures from the work.

#### Training and Test Metrics

Training procedure was tracked using [Weights & Biases](https://wandb.ai/site) platform and can be viewed in [the associated public project](https://wandb.ai/alsk/lldqn?workspace=user-alsk). The same platform was also used to generate reported plots. If you do your own training, the logging infrastructure is already setup and you should only pass-in your own [API key](https://docs.wandb.ai/quickstart#1.-set-up-wandb).

#### Task Similarity

The task similarity confusion matrix was generated by the following command:

```
python src/scripts/evaluate_autoencoder.py
```

The axis labels were added manually, to aid the visibility.

#### Custom Evaluation

These results were not reported directly, but were used during development for my own reference. Nevertheless, I include them here for completeness. To run a test evaluation for both, baseline and LLDQN policies, you can run:

```
python src/scripts/evaluate_lldqn.py
python src/scripts/evaluate_baseline.py
```

Both scripts will log their output to a specified W&D project. Feel free to customize the environments and other configuration options.
