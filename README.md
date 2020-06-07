# classifier

To run segmentation pipeline 
 1. git clone https://github.com/LE-V-EL/classifiers.git
 2. conda env create -f env.yml
 3. conda activate env_name	
 4. get the npz files to local 
 5. python3 segmentation.py dataset_name npz_path number_of_epochs history_file_path
 
 For example:
    python3 segmentation.py angle /home/aswin/Desktop/output 1 /history
 
 
