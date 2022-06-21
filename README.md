# Code for the models in the paper: Predicting Aqueous Solubility of Organic Molecules Using Deep Learning Models with Varied Molecular Representations (https://pubs.acs.org/doi/full/10.1021/acsomega.2c00642)

#### Usage
1. Download data from https://figshare.com/s/542fb80e65742746603c and save it as data.csv in the ./data folder
2. Generate Pybel coordinates and MDM features by running create_data.py at the ./data folder
3. To train MDM, GNN, SMI and SCH models run train.py at ./mdm, ./gnn, ./smi and ./sch folderes respectively.
4. To make predictions use predict.ipynb files at each model folders.

Best models used to obtain the results in the paper are included in each model folders (mdm_paper.h5, gnn_paper.pt, smi_paper.h5, sch_paper.pt)

---------------------------------------------------------------------------------------------------
</br></br>
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
<center>PACIFIC NORTHWEST NATIONAL LABORATORY</center>
<center>operated by</center>
<center>BATTELLE</center>
<center>for the</center>
<center>UNITED STATES DEPARTMENT OF ENERGY</center>
<center>under Contract DE-AC05-76RL01830</center>

