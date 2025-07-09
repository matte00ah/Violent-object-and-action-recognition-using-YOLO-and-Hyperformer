import os


def write_skel_available_name(path, skels_path): 
    file = open(path+"/skes_available_name.txt", 'a')
    for f in os.listdir(skels_path):
        if f.endswith(".skeleton"):
            file.write(f[:-9]+"\n")
    file.close()

def write_setup(path, skels_path):
    file = open(path+"/setup.txt", 'a')
    for f in os.listdir(skels_path):
        if f.endswith(".skeleton"):
            file.write("1\n")
    file.close()

def write_performer(path, skels_path):
    file = open(path+"/performer.txt", 'a')
    for f in os.listdir(skels_path):
        if f.endswith(".skeleton"):
            file.write("1\n")
    file.close()

def write_label(path, skels_path):
    file = open(path+"/label.txt", 'a')
    for f in os.listdir(skels_path):
        if f.endswith(".skeleton"):
            file.write(str(int(f[-12:-9]))+"\n")
    file.close()

if __name__ == '__main__':
    path = "/work/cvcs2024/ViolenceDatasetHyperformer/FTD_processed/statistics"
    skels_path = "/work/cvcs2024/ViolenceDatasetHyperformer/FineTuning-dataset/skeletons"
    write_skel_available_name(path, skels_path)
    write_setup(path, skels_path)
    write_performer(path, skels_path)
    write_label(path, skels_path)