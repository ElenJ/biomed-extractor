# This dataset for training was used from https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/, the AD and EBM-NLPmod dataset. 
import os


if __name__ == "__main__":
    PROJECT_ROOT1 = os.path.expanduser('~/Documents/github/section_specific_annotation_of_PICO-main/data/AD')
    PROJECT_ROOT2 = os.path.expanduser('~/Documents/github/section_specific_annotation_of_PICO-main/data/EBM-NLPmod')
    PROJECT_ROOTS = [PROJECT_ROOT1, PROJECT_ROOT2]
    OUTPUT_FOLDER = os.path.expanduser('~/Documents/github/biomed_extractor/data/pico_dataset_for_training')
    # Data directory at top level
    for PROJECT_ROOT in PROJECT_ROOTS:
        folders = os.listdir(PROJECT_ROOT)
        print(folders)  
        for folder in folders:
            folder_path = os.path.join(PROJECT_ROOT, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith(".txt"):                        
                        print(f"Processing file: {file_path}")
                        # read dev, test, train.txt files and merge them into one file
                        with open(file_path, "r") as file:
                            lines = file.readlines()
                        #with open(os.path.join(OUTPUT_FOLDER, f"PICO_merged.txt"), "a") as outfile:
                        #    outfile.writelines(lines)
                    if file_name.endswith("train.txt"):
                        with open(file_path, "r") as file:
                            lines = file.readlines()
                        with open(os.path.join(OUTPUT_FOLDER, f"PICO_merged_train.txt"), "a") as outfile:
                            outfile.writelines(lines)
                    if file_name.endswith("test.txt"):
                        with open(file_path, "r") as file:
                            lines = file.readlines()
                        with open(os.path.join(OUTPUT_FOLDER, f"PICO_merged_test.txt"), "a") as outfile:
                            outfile.writelines(lines)
                    if file_name.endswith("dev.txt"):
                        with open(file_path, "r") as file:
                            lines = file.readlines()
                        with open(os.path.join(OUTPUT_FOLDER, f"PICO_merged_dev.txt"), "a") as outfile:
                            outfile.writelines(lines)

