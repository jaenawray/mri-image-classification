import os
import shutil

input_folder = '/Users/mayasachidanand/Documents/spring25/cs4100/processed_mat'
output_folder = '/Users/mayasachidanand/Documents/spring25/cs4100/processed_mat/renamed_files'

classification_dict = {'1': 'me', '2': 'gl', '3': 'pi'}

for root, dirs, files in os.walk(input_folder):
    folder_name = os.path.basename(root) # starting from the second loop, root will be '1', '2', '3'

    if folder_name in classification_dict: # first loop will not since the folder is 'processed_mat'
        tumor_type = classification_dict[folder_name]

        for file in files:
            if file.endswith('.jpg'):
                new_name = f'{tumor_type}_{file}'

                file_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                #shutil.copy(file_path, dest_path)
                os.rename(file_path, new_path) 

