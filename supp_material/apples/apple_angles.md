code expression for all 50 angle values
---------------------------------------
matlab: `linspace(0.5*pi/50., pi-0.5*pi/50., 50)`
python: `odl.uniform_discr(0., np.pi, 50).meshgrid[0]`


50 angles
---------
matlab index: `1:1:50`
python index: `0:50:1`
angles: `[0.03141593, 0.09424778, 0.15707963, 0.21991149, 0.28274334, 0.34557519, 0.40840704, 0.4712389, 0.53407075, 0.5969026, 0.65973446, 0.72256631, 0.78539816, 0.84823002, 0.91106187, 0.97389372, 1.03672558, 1.09955743, 1.16238928, 1.22522113, 1.28805299, 1.35088484, 1.41371669, 1.47654855, 1.5393804, 1.60221225, 1.66504411, 1.72787596, 1.79070781, 1.85353967, 1.91637152, 1.97920337, 2.04203522, 2.10486708, 2.16769893, 2.23053078, 2.29336264, 2.35619449, 2.41902634, 2.4818582, 2.54469005, 2.6075219, 2.67035376, 2.73318561, 2.79601746, 2.85884931, 2.92168117, 2.98451302, 3.04734487, 3.11017673]`

10 angles
---------
matlab index: `3:5:50`
python index: `2:50:5`
angles: `[0.15707963, 0.4712389, 0.78539816, 1.09955743, 1.41371669, 1.72787596, 2.04203522, 2.35619449, 2.67035376, 2.98451302]`

5 angles
--------
matlab index: `6:10:50`
python index: `5:50:10`
angles: `[0.34557519, 0.97389372, 1.60221225, 2.23053078, 2.85884931]`

2 angles
--------
matlab index: `13:25:50`
python index: `12:50:25`
angles: `[0.78539816, 2.35619449]`


Motivation for choosing these subsets
-------------------------------------
The original 50 angles are chosen as in `odl.uniform_discr(0., np.pi, 50).meshgrid[0]`.
The first angle here is at "half an angle step size", and the last angle is at "pi minus half an angle step size".
For 10 and for 2 angles, the angles subset matches with this convention, i.e. `odl.uniform_discr(0., np.pi, 10).meshgrid[0]` or `odl.uniform_discr(0., np.pi, 2).meshgrid[0]` respectively.
For 5 angles, the subset cannot be chosen in this manner, but needs to be shifted by half an angle step. Here we choose to shift in positive direction.
