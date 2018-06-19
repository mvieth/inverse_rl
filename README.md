# Inverse RL

Implementations for imitation learning / IRL algorithms in RLLAB

Contains:
- GAIL (https://arxiv.org/abs/1606.03476/pdf)
- Guided Cost Learning, GAN formulation (https://arxiv.org/pdf/1611.03852.pdf)
- Tabular MaxCausalEnt IRL (http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

Setup
---
This library requires:
- rllab (https://github.com/openai/rllab)
- Tensorflow

Pendulum example
---

Running the Pendulum-v0 gym environment:

1) Collect expert data
```
python scripts/pendulum_data_collect.py
```

You should get an "AverageReturn" of around -100 to -150

2) Run imitation learning
```
python scripts/pendulum_irl.py
```

The "OriginalTaskAverageReturn" should reach around -100 to -150

3) Run GAIL as a comparison
```
python scripts/pendulum_gail.py
```

Ant example
---

1) Collect expert data
```
python scripts/ant_data_collect.py
```

2) Run AIRL
```
python scripts/ant_irl.py
```

3) Transfer to a disabled ant using the parameters from step 2
```
python scripts/ant_transfer_disabled.py
```
