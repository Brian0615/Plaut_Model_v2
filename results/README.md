# Results

### LENS-20210516
 * **Description**: Analysis on distance between anchors and probes for the newly trained attractor network on LENS
 * Please refer to the Jupyter notebook files inside this folder for the analysis results

### PLAUTFIG18-20200907
 * **Description**: Six sets of simulations with varying parameters for replicating Fig 18 in the Plaut paper
 * **Test Descriptions**:
   * TEST1: Base simulation
   * TEST2: Base simulation with increased weight decay
   * TEST3: Base simulation with further increased weight decay
   * TEST4: SGD optimizer only (lr=0.0001, m=0.9)
   * TEST5: SGD optimizer only with increased weight decay
   * TEST6: SGD optimizer only with further increased weight decay
 * **Notes**: Please see `Correlation Plots.docx` for plots and more detailed descriptions of tests.

### BASE-20200818
 * **Description**: Change loss function to generalized cross entropy loss
 * **Note**: GCE loss will be used for all future simulations unless noted otherwise

### BASE-20200718
 * **Description**: Simulations with target radius of 0.1 added
 * **Note**: Target radius will be used for all future simulations unless noted otherwise
