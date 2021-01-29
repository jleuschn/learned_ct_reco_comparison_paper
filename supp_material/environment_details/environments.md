The environment `main` was used for most trainings.
To use the cINN models, additional packages for `main` are required. These are listed in the `cinn` environment.
For training the mixed-scale dense network, a separate environment `msd_pytorch` was used.
The files `main_env.txt` and `msd_pytorch_env.txt` contain snapshots of the lists of installed packages in these environments.

To create similar environments, the following commands can be used (at the time of writing, 2020-11-11).

###### main
```
conda create -n "main" pytorch tensorboard astra-toolbox -c pytorch -c defaults -c astra-toolbox/label/dev
pip install https://github.com/odlgroup/odl/archive/master.zip https://github.com/ahendriksen/tomosipo/archive/develop.zip dival
```

The tensorboard package is optional (used for training logs).

###### cinn
Based on main. Install additional packages via
```
pip install pytorch-lightning torchvision git+https://github.com/VLL-HD/FrEIA.git
```

###### msd_pytorch
```
conda create -n "msd_pytorch" msd_pytorch=0.9.0 cudatoolkit=10.1 torchvision tensorboard astra-toolbox -c aahendriksen -c pytorch -c defaults -c conda-forge -c astra-toolbox/label/dev
pip install https://github.com/odlgroup/odl/archive/master.zip https://github.com/ahendriksen/tomosipo/archive/develop.zip dival
```

The tensorboard package is optional (used for training logs).

###### ictunet
See requirements at this [github repo](https://github.com/DomiBa/Apples-CT-challenge).
