import os
from tqdm import tqdm

import shutil


class DataSplitter():
    def __init__(self, list_dirs_to_datasets, out_dir_path) -> None:
        self.listdir_in = list_dirs_to_datasets # Path to directories containing subdirectories of Train/Validation/Test
        self.path_to_train = [os.path.join(self.listdir_in[i], "Train") for i in range(len(self.listdir_in))]
        self.path_to_validation = [os.path.join(self.listdir_in[i], "Validation") for i in range(len(self.listdir_in))]
        self.path_to_test = [os.path.join(self.listdir_in[i], "Test") for i in range(len(self.listdir_in))]
        self.class_names_train = [cname for cname in os.listdir(self.path_to_train[0]) if os.path.isdir(os.path.join(self.path_to_train[0], cname))]
        self.class_names_val = [cname for cname in os.listdir(self.path_to_validation[0]) if os.path.isdir(os.path.join(self.path_to_validation[0], cname))]
        self.class_names_test = [cname for cname in os.listdir(self.path_to_test[0]) if os.path.isdir(os.path.join(self.path_to_test[0], cname))]
        self.train_samples_per_dataset = [len([file for file in os.listdir(os.path.join(train_dir, self.class_names_train[0])) if file.endswith('.fits')]) for train_dir in self.path_to_train]
        self.validation_samples_per_dataset = [len([file for file in os.listdir(os.path.join(val_dir, self.class_names_val[0])) if file.endswith('.fits')]) for val_dir in self.path_to_validation]
        self.test_samples_per_dataset = [len([file for file in os.listdir(os.path.join(test_dir, self.class_names_test[0])) if file.endswith('.fits')]) for test_dir in self.path_to_test]
        
        # Check if output directory exists:
        self.out_train_path = os.path.join(out_dir_path, "Train")
        self.out_validation_path = os.path.join(out_dir_path, "Validation")
        self.out_test_path = os.path.join(out_dir_path, "Test")
        
        ## Train, Validation, Test Main directories
        if not os.path.isdir(self.out_train_path):
            os.mkdir(self.out_train_path)
        if not os.path.isdir(self.out_validation_path):
            os.mkdir(self.out_validation_path)
        if not os.path.isdir(self.out_test_path):
            os.mkdir(self.out_test_path)

        ## Class-Directories
        class_names_dirs = [self.class_names_train, self.class_names_val, self.class_names_test]
        for i, maindir in enumerate([self.out_train_path, self.out_validation_path, self.out_test_path]):
            for fname in class_names_dirs[i]:
                if not os.path.isdir(os.path.join(maindir, fname)):
                    os.mkdir(os.path.join(maindir, fname))

        print("Directory Tree Created!")
    def Run(self):

        glob_train_file_idx = 0
        glob_validation_file_idx = 0
        glob_test_file_idx = 0

        # Merge Training directories
        for idx, dir_to_train_dataset in tqdm(enumerate(self.path_to_train), desc="Merging Training sub-directories..."):
            for fnum in range(self.train_samples_per_dataset[idx]):
                for cnum, fname in enumerate(self.class_names_train):
                    source = os.path.join(dir_to_train_dataset, os.path.join(self.class_names_train[cnum],f"{fname}_{fnum}.fits"))
                    dest = os.path.join(os.path.join(self.out_train_path, fname), f"{fname}_{glob_train_file_idx}.fits")
                    shutil.move(source, dest)

                glob_train_file_idx += 1

        # Merge validation directories
        for idx, dir_to_validation_dataset in tqdm(enumerate(self.path_to_validation), desc="Merging Validation sub-directories..."):
            for fnum in range(self.validation_samples_per_dataset[idx]):
                for cnum, fname in enumerate(self.class_names_val):
                    source = os.path.join(dir_to_validation_dataset, os.path.join(self.class_names_val[cnum],f"{fname}_{fnum}.fits"))
                    dest = os.path.join(os.path.join(self.out_validation_path, fname), f"{fname}_{glob_validation_file_idx}.fits")
                    shutil.move(source, dest)

                glob_validation_file_idx += 1

        # Merge Test directories
        for idx, dir_to_test_dataset in tqdm(enumerate(self.path_to_test), desc="Merging Test sub-directories..."):
            for fnum in range(self.test_samples_per_dataset[idx]):
                for cnum, fname in enumerate(self.class_names_test):
                    source = os.path.join(dir_to_test_dataset, os.path.join(self.class_names_test[cnum],f"{fname}_{fnum}.fits"))
                    dest = os.path.join(os.path.join(self.out_test_path, fname), f"{fname}_{glob_test_file_idx}.fits")
                    shutil.move(source, dest)

                glob_test_file_idx += 1

        print("Data Sucessfully Split!")

    def Clean(self):
        for old_dir in self.listdir_in:
            shutil.rmtree(old_dir)
        print("Directory Cleaned!")


base = "/mnt/d/SPIRE-SR-AI/data/processed/dummy_set" #r"/scratch-shared/dkoopmans/120deg2_shark_sides"
SHARK = [f"SHARK_{i+1}" for i in range(0, 2)]
SIDES = [f"SIDES_{i+1}" for i in range(0, 2)]
datasets = SHARK + SIDES # Prefixes of the datamaps. Check the code for "fname" for details on standard formatting of files. CTRL + F --> "fname"

listdir_to_datasets = [os.path.join(base, dataset) for dataset in datasets]

splitter = DataSplitter(listdir_to_datasets, base)
splitter.Run()
splitter.Clean()