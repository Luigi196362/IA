import os

dataset_dir = 'dataset'

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for idx, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1]
            new_name = f"{folder}_{idx:02d}{ext}"
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)
            os.rename(src, dst)