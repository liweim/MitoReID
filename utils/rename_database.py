import os
import shutil
import pandas as pd
import glob
import math
import os.path as osp


def get_id(info, layout_table, path, plate_name):
    ps = path.replace("_s1", "").split("_")
    chid = ps[1]
    w = "".join(ps[2:])[:2]
    ch_id = int(chid[1:])
    ch = chid[0]

    if "01_w" in path:
        drug_id = "ctrl"
    else:
        fda_index = info.index(plate_name)
        ch_index = fda_index + info.index(ch)
        drug_id = layout_table.iloc[ch_index, ch_id]

    return drug_id, chid, w


def refactor(data_path, drug_path, save_path, plate_name):
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)

    layout_table = pd.read_excel(drug_path)
    info = layout_table.iloc[:, 1].values.tolist()

    for path in glob.glob(data_path + "/*/*.tif"):
        if "_thumb" in path:
            continue

        folder, filename = os.path.split(path)
        time = os.path.split(folder)[1].split("_")[1]
        drug_id, chid, w = get_id(info, layout_table, filename, plate_name)
        if type(drug_id) is not str and math.isnan(drug_id):
            continue

        if type(drug_id) is not str:
            drug_id = str(int(drug_id))
        new_filename = f"{drug_id}/{drug_id}_{time}_{w}.tif"
        new_path = os.path.join(save_path, new_filename)
        while os.path.exists(new_path):
            wrong = os.path.split(os.path.split(new_path)[0])[1]
            if "-" in wrong:
                tmp, num = wrong.split("-")
                correct = f"{tmp}-{int(num) + 1}"
            else:
                correct = f"{wrong}-2"
            new_filename = f"{correct}/{correct}_{time}_{w}.tif"
            new_path = os.path.join(save_path, new_filename)
        dir = os.path.split(new_path)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        print(path, new_path)
        shutil.copy(path, new_path)


if __name__ == "__main__":
    plate_layout_path = "F:/20230911/20230911.xlsx"

    data_path = "F:/20230911\YM-20230911-60PA-HJTG-2-3_Plate_191_192_193"
    save_path = data_path+" rename"
    plate_name = "plate_191+192+193"
    refactor(data_path, plate_layout_path, save_path, plate_name)

