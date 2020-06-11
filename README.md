# classifier

To run segmentation pipeline 
 1. git clone https://github.com/LE-V-EL/classifiers.git
 2. git submodule init
 3. conda env create -f env.yml
 4. conda activate env_name	
 5. get the npz files to local 
 6. python3 segmentation.py dataset_name npz_path number_of_epochs history_file_path gpu
 
 For example:
    python3 segmentation.py angle /home/aswin/Desktop/output 1 /history 0 
 
 
