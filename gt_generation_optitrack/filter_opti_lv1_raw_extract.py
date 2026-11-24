import numpy as np
import os
import os.path as osp
import argparse
import datetime
from joints_order import j_order

class CSV_Reader(object):
    def __init__(self, num_person, file_path, output_dir):
        self.csv_lines = None
        self.file_path = file_path
        seq = file_path[file_path.rfind("/")+1:].replace("_raw.csv","")
        print(seq)
        self.save_path = osp.join(output_dir, seq, f"gt/{seq}.csv")
        # print(self.save_path)
        # exit(0)
        self.num_person = num_person
        with open(self.file_path, "r") as f:
            self.csv_lines = f.read().split("\n")[:-1]
        
        self.start_ts = self.get_timestamp() # in second

        if (seq in ["civit_easy_01","civit_norm_01","civit_norm_02","civit_norm_03",
                    "civit_norm_04","civit_norm_05"]):
            self.type_list = ["Marker", "Rigid Body"]
        else:
            self.type_list = ["Bone Marker", "Rigid Body"]
        self.prefer_order = list(j_order.keys()) + ['SensorBox']
        
        self.get_data()

    def get_timestamp(self):
        line = self.csv_lines[0].split(",") # First line has timestamp
        ts_idx = line.index("Capture Start Time") + 1
        ts_line = line[ts_idx]

        ts_date = ts_line[:19] # Get date with format %Y-%m-%d %H.%M.%S
        element = datetime.datetime.strptime(ts_date,"%Y-%m-%d %H.%M.%S")
        timestamp = float(datetime.datetime.timestamp(element))
        ts_ms   = float(ts_line[20:-4])
        adj_s = 43200 if ts_line[-3:] == "IP." else 0

        timestamp += ts_ms * 1e-3 + adj_s

        return timestamp

    def get_data(self):
        type_line = self.csv_lines[2].split(",") # Bone - Bone Marker - Rigid Body - ...
        name_line = self.csv_lines[3].split(",") # Skeleton001: --....
        id_line   = self.csv_lines[4].split(",") # 1 - 2 - 3 - ....
        header    = self.csv_lines[6].split(",") # Frame - Time (Seconds) - X - Y - Z - ...
        data      = self.csv_lines[7:]

        # Get selected type index
        type_idx  = [0,1] + [i for i, x in enumerate(type_line) if x in self.type_list
                                        and "SensorBox:Marker" not in name_line[i]]
        update_idx = [0,1]
        for person_id in self.num_person:
            prefix = "Skeleton 00" + str(person_id) + ":"
            for prefer in self.prefer_order[:-1]:
                for idx in type_idx:
                    pref = prefix + prefer
                    if (pref in name_line[idx]):
                        update_idx.append(idx)
        for idx in type_idx:
            pref = "SensorBox"
            if (pref in name_line[idx]):
                update_idx.append(idx)

        type_idx = update_idx

        # Process name_line with prefer_order
        # prefer_index = [index for prefer in self.prefer_order[]] 
        header_out = ",".join([n+'+'+h for n, h in zip(np.asarray(name_line)[type_idx], 
                                                       np.asarray(header)[type_idx])])
        
        os.makedirs(osp.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            f.write(header_out+"\n")
            for data_line in data:
                data_split = data_line.split(",")
                
                # Adjust timestamp
                time = (self.start_ts + float(data_split[1]))*1e9
                time = "{:.0f}".format(time)
                data_split[1] = time

                data_out = ",".join(np.asarray(data_split)[type_idx])
                f.write(data_out+"\n")
        
        # with open(self.file_path[:-4] + "_stage_02.csv", "w") as f:

        

def main():
    parser = argparse.ArgumentParser(
        prog="filter_optitrack_step1.py",
        description="Process raw optitrack into data point with timestamp and index, \
                     in world coord"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        required=True,
        help="Path to raw csv file directory"
    )

    # parser.add_argument(
    #     "-i", "--input_file",
    #     type=str,
    #     help="Path to raw csv file"
    # )

    # parser.add_argument(
    #     "-o", "--output_file",
    #     type=str,
    #     help="Path to save refine csv file"
    # )

    parser.add_argument(
        "--sel",
        default="",
        help="Select sequence to process instead of processing all sequences"
    )

    parser.add_argument(
        "-n", "--num_person",
        type=int,
        default=1,
        help="Number of person in the recording"
    )

    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory containing the extracted sequences")

    args = parser.parse_args()

    # select_files = ["civit_easy_01_raw"]
    select_files = []
    if (args.sel != ""):
        select_files = [args.sel]

    if (len(select_files)==0):
        files = sorted(os.listdir(args.dir))
    else:
        files = select_files
    for file in files:
        if ("csv" not in file):
            print(f"{file} does not contain .csv")
            continue
        
        n_person = [1]
        if ("multi" in file):
            n_person = [1, 2]
            if ("06" in file or "07" in file):
                n_person = [2, 3]
        if ("prashant" in file):
            n_person = [2]
        if ("nikolay" in file):
            n_person = [3]
        csv_reader = CSV_Reader(n_person, osp.join(args.dir, file), args.output_dir)

if __name__ == "__main__":
    main()