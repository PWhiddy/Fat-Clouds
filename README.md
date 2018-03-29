# 3D Fluid

A bare-bones but polished fluid simulator and volumetric renderer written in CUDA C/C++ in ~600 LOC.

**There are** [**videos**](https://drive.google.com/drive/folders/1A8PH2aZoj2ab8UDZKSdC6fsMPB5RUtyr?usp=sharing)!

The organization is based on Philip Rideout's 2D OpenGL simulator (http://prideout.net/blog/?p=58)

As well as George Corney's interactive WebGL demo (https://github.com/haxiomic/GPU-Fluid-Experiments)

TODO: More consolidation of simulation and state parameters. Temperature-based shading. Support for data types smaller than 4 byte floats.

# Setup

In ```build.sh```, check that the path to your cuda installation is correct. Then run ```build.sh``` to create an executable.

![](https://i.imgur.com/qKtCdZf.png "Render")

Navier Tokes yo

![](https://i.imgur.com/uYr2u7y.png "Render")

![](https://i.imgur.com/Y3MGgck.png "Render")

![](https://i.imgur.com/g8OmfZA.png "Render")

![](https://i.imgur.com/dMWps1a.png "Render")

Testing the renderer with spheres

![](https://i.imgur.com/un8Smjb.jpg "Render")
