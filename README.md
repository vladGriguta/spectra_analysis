
## Run on GPU

To run a python script using the provided container first ssh to a LOFAR machine you have booked time on. 
Ensure you are in bash. (just type bash). In order to use singularity on the LOFAR machines you must add it to your path:

```bash
export PATH=/raid/singularity/bin:$PATH
```

To download the container to your working directory run


```bash
singularity pull --name container.simg shub://JBCA-MachineLearning/GPUs:cuda10
singularity pull --name CPU_container.simg shub://vladGriguta/GPUs:cuda10
```

From your working directory run the following command
```bash
export SINGULARITY_BINDPATH=$PWD:/mnt
```

This will bind your working directory to the folder /mnt in the singularity container when you run the container. 

To run your python script use the following command

```bash
singularity run --nv -H /raid/scratch/vladg/spectraClassification/spectra_analysis/ GPU_container.simg /mnt/current_script.py
```
