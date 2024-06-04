# Used to generate csv, which contains the pairs of patch path and label
def index_files(directory, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'label'])
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    path = os.path.join(root, file)
                    label = path.split("\\")[-3]
                    writer.writerow([path, label])

# tackle train or test samples respectively
index_files('the path of train/test patches', 'the path of output csv')
