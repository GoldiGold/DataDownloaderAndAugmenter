import os, sys
import json
import consts


def get_dataset_names(data_dir: str = consts.DATASET_DIR):
    '''
    This function gets the names (or ids) of the brains in the dataset and returns a list of the ids (in a bash script
    format) of them
    :param data_dir: the data directory that holds the dataset and tables
    :return: the list it self
    '''
    os.chdir(data_dir)
    dirs = os.listdir()
    # print(dirs)
    dataset_names = []
    count = 0
    for dir in dirs:
        if os.path.isdir(dir):
            count += 1  # confirmed the correct amount of directories in the Dataset.
            dataset_names.append(int(dir))
    return count, dataset_names


def compare_tables(table_dir: str):
    '''
    Thus function checks to see if the labels file that was generated from the apac+aseg.nii.gz files are the same
    :param table_dir: the directory of the table files
    :return: True and 0 if they are the same, False and the file id's that broke the similarity
    '''
    files = os.listdir(table_dir)
    diff_files = []
    with open(os.path.join(table_dir, files[0]), 'r') as first_file:
        first = first_file.read()
    print(f'first file is {files[0]}')
    for f in files[1:]:
        with open(os.path.join(table_dir, f), 'r') as some_file:
            some = some_file.read()
        if len(first) == len(some):
            if first != some:
                print("isn't the same value")
                diff_files.append(f)
        else:
            # print(f"isn't the same length, first: {len(first)} some: {len(some)}")
            diff_files.append(f[consts.START_ID:consts.END_ID])
    return len(diff_files) == 0, diff_files


def check_if_labels_exists(labels_names: list, table_dir: str):
    '''
    This function checks if the labels stated in the list exist in all the table files in the directory.
    :param labels_names: the labels
    :param table_dir: the table directory
    :return: True, empty dictionary if labels exist in all files. False, dictionary with files:missing labels as the key:value
    '''
    files = os.listdir(table_dir)
    missing = {}

    for f in files:
        with open(os.path.join(table_dir, f), 'r') as some_file:
            some = some_file.read()
        for l in labels_names:
            if l not in some:
                if f[consts.START_ID:consts.END_ID] not in missing.keys():
                    missing[f[consts.START_ID:consts.END_ID]] = []
                missing[f[consts.START_ID:consts.END_ID]].append(l)
    return len(missing) == 0, missing


def create_brain_labels_json(labels_names: list, brain_table_dir: str, brain_id: int):
    info = {}  # {"id": brain_id}
    for l in labels_names:
        info[l] = None
    with open(f'{brain_table_dir}', 'r') as some_file:
        some = some_file.read()
    for l in labels_names:
        index = some.find(l)
        if index != -1:
            space_index = some[index + len(l) + 1:].find(' ')  # the +1 is for the \n
            # print(
            #     f"Start:{some[index + len(l) + 1:index + len(l) + 1 + space_index]},{int(some[index + len(l) + 1:index + len(l) + 1 + space_index])}:end")
            info[l] = int(some[index + len(l) + 1:index + len(l) + 1 + space_index])
            # if f[START_ID:END_ID] not in missing.keys():
    return info  # json.dumps(info)


def create_all_brains_labels_json(labels_names: list, table_dir: str, brain_ids: list, json_file: str):
    all_brains_dict = {}
    with open(json_file, 'w', encoding='utf-8') as f:
        for table in os.listdir(table_dir):
            brain_id = int(table[consts.START_ID:consts.END_ID])
            # for brain_id in brain_ids:
            brain_dict = create_brain_labels_json(labels_names, os.path.join(table_dir, table), brain_id)
            all_brains_dict[brain_id] = brain_dict
            # print(brain_dict)

        json.dump(all_brains_dict, f, ensure_ascii=False, indent=4)


def main(data_dir: str = consts.DATA_DIR, wb_dir: str = consts.WB_DIR):
    '''
    This function was supposed to create the table files from the aparc+aseg files but wb_command can't be called with
    os.system so we created a bash script that does that
    :param data_dir: the data directory that holds the dataset and tables
    :param wb_dir: the directory to get the wb_command command
    :return: count - the amount of tables it created
    '''
    os.chdir(wb_dir)
    # print(os.listdir())
    # return 0
    dirs = os.listdir(os.path.join(data_dir, 'Dataset'))
    # print(dirs)
    count = 0
    for dir in dirs:
        if os.path.isdir(os.path.join(data_dir, 'Dataset', dir)):
            count += 1  # confirmed the correct amount of directories in the Dataset.
            os.system(
                f'wb_command -volume-label-export-table {data_dir}Dataset/'
                f'{dir}/MNINonLinear/aparc+aseg.nii.gz 1 {data_dir}Tables/output-table-{dir}.txt')
            # print(f'this is a dir {dir}')
            # return 'be happy'
    print(count)
    return count


if __name__ == '__main__':
    _, names = get_dataset_names()
    # lst = str(names)
    # lst = lst[1:-1].replace(',', '')
    # print(lst)
    create_all_brains_labels_json(consts.LABELS, os.path.join(consts.DATA_DIR, 'Tables'), names,
                                  '/home/cheng/PycharmProjects/DataDownloaderAndAugmenter/labels.json')
    j = create_brain_labels_json(consts.LABELS, os.path.join(consts.DATA_DIR, 'Tables', 'output-table-100307.txt'),
                                 100307)
    print(j)

    # eq, miss = check_if_labels_exists(LABELS, f'{DATA_DIR}Tables/')
    # print(eq, miss)
    # eq, ids = compare_tables(f'{DATA_DIR}Tables/')
    # print(eq, len(ids), ids)
