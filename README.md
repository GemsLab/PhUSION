# Node Proximity Is All You Need:Unified Structural and Positional Node and Graph Embedding
This is a reference implementation for PhUSION, a proximity-based unified framework for computing structural and positional node embeddings, which leverages well-established methods for calculating node proximity scores.

To run proximity/structural embedding codes:
``` python src/main.py --param parameters/structural/PPMI.json ```  
To run graph classification codes under graph_embed directory
``` python main.py --param ../../parameters/graph/PPMI.json ```  
  
*input parameters*  
- All the parameters are stored in a single .json file(examples are in the parameter/ directory), which contains 6 fields:  
	* "input": path of input file(.mat file)  
	* "prox_option": "FaBP", "heat_kernel" or "netmf"  
	* "prox_params": dict of parameters needed.  
		* transform indicates which nonlinear transform you would like to use, 0 is no nonlinear transform, 1 stands for log transform and 2 for binary threshold
		* threshold indicates for bineary transform, which threshold you would like to use  
	* "prox_file": filename of the intermediate proximity matrix copy
	* "embed_option": "proximity" or "struct"  
	* "embed_params": dict of parameters needed.  
		* For proximity: dim  
		* For struct: time_pnts  
	* "output": path for output file(.npy)  

*data directory*
- Include three subdirectories:  
	* origin: store the graphs(.mat file)  
	* proximity: store the intermediate data(.mat file, proximity matrix)
	* embeded: store the embeded matrix(.npy file)

*src directory*
- Now different methods are stored in different .py files(PPMI is stored in src/proxi_methods/PPMI.py)  
- eval subdirectory contains predict.py and dist.py. 
	* Run dist.py to eval the proximity matrix  
	example:  
	``` python3 src/eval/dist.py --input data/proximity/heat_kernel_struct.mat```
	* Run predict.py to evaluate performance  
	example:  
	``` python src/eval/predict.py --input data/origin/usa-airports.edgelist --embedding data/embeded/structural/PPMI.npy --seed 0```
