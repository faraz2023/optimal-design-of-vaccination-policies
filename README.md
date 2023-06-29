# DCNDP_IP
GitHub repository for work **Faraz Khoshbakhtian, Hamidreza Validi, Mario Ventresca, Dionne
Aleman. Optimal design of vaccination policies: A case study for Newfoundland and Labrador**.

## Setting up the environment

This code base is written in Python 3.10 and leverages Gurobi 10.0.1 for solving IP problems. 
To set up the environment, make sure Gurobi is installed and licensed on your system. 

You can use the following command to install the required libraries:

```
pip install -r requirements.txt
```

In case pip installation did not work for the NetworkX-metis library, follow the steps here to install the package from the source: https://networkx-metis.readthedocs.io/en/latest/install.html 


## Running the experiments

This repository includes implementations for 1-hop and 2-hop DCNDP experiments. We provide a pipeline for running the IP models on simulated networks of the NL province with more than 500K nodes. This pipeline uses node attributes (e.g., Health Authority responsible for the individuals) and network topology to divide the network into smaller partitions for IP optimization. We leverage the well-known METIS algorithm for topologically partitioning the network.


## 2-hop experiments

Run: 

```
# First partitions the network using dividing_col (here health authority), then partition via METIS 
python DCNDP_2hop_Partition_Regional.py --export_path DCNDP_NL_D2_2hop_healthAuthority_ID --input_path DCNDP_Datasets/NL_Day2 --dividing_col healthAuthority_ID --partition_avg_size 2500 --partition_remove_budget 0.2 --timelimit 3600

# Run with disaggregated constraints. First partitions the network using dividing_col (here health authority), then partition via METIS
python DCNDP_2hop_Partition_Regional.py --export_path DCNDP_NL_D2_2hop_healthAuthority_ID_disagg --input_path DCNDP_Datasets/NL_Day2 --dividing_col healthAuthority_ID --partition_avg_size 2500 --partition_remove_budget 0.2 --timelimit 3600 --disaggragated_constraints 
```

```
# Run with raw METIS partitions
python DCNDP_2hop_Partition_Raw.py --export_path DCNDP_NL_D2_2hop_raw --input_path DCNDP_Datasets/NL_Day2 --num_partition 200 --partition_remove_budget 0.2 --timelimit 3600

# Run with disaggregated constraints and with raw METIS partitions
python DCNDP_2hop_Partition_Raw.py --export_path DCNDP_NL_D2_2hop_raw_disagg --input_path DCNDP_Datasets/NL_Day2 --num_partition 200 --partition_remove_budget 0.2 --timelimit 3600 --disaggragated_constraints

```


## 1-hop experiments

Run:

```
# First partitions the network using dividing_col (here health authority), then partition via METIS
python DCNDP_1hop_Partition_Regional.py --export_path DCNDP_NL_D2_1hop_healthAuthority_ID --input_path DCNDP_Datasets/NL_Day2 --dividing_col healthAuthority_ID --partition_avg_size 25000 --partition_remove_budget 0.2 --timelimit 3600 
```

```
python DCNDP_1hop_Partition_Raw.py --export_path DCNDP_NL_D2_1hop_raw --input_path DCNDP_Datasets/NL_Day2 --num_partition 20 --partition_remove_budget 0.2 --timelimit 3600
```
