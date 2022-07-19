# Universal-image-segementation user guide
Segmentation program to identify the thickness of 2D materials from optical microscopy images. 
# Training
# 1. Open the file layerID_train_new.py 
# 2. Enter the parameters and run the program (refer to the list of parameters for details on the information to give for each parameter)
# 3. For the img_file use an image of a sample of which the layers have already been identified
#   - Note that when cropping the image the y-values increase going from top to bottom
#   - The out_file should be an empty/nonexistent file. It will be filled/created when the code is run (remember to change this parameter for each new material/substrate combination)
# 4. When the processed image loads, write down the thicknesses and number of each cluster
#   - The cluster number is found by hovering the mouse over a cluster and looking at the number in brackets at the bottom right of the figure (these numbers are random and are not yet correlated to the cluster thickness)
# 5. Close the figures and enter the cluster thicknesses as prompted

# Master catalog and testing
# 1. Open master_cataloger.py 
# 2. Enter the parameters and run the program
# 3. Open layerID_test_new.py
# 4. Enter the parameters and run the program 
#   - To test the program use an image with known thicknesses, otherwise the image can be one where the thicknesses are unknown
# 5. The output will be a processed image that identifies the thickness of each layer of a given flake
