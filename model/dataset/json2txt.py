import json
import os
from tqdm import tqdm
import glob
import os.path as osp

# Reference: https://blog.csdn.net/m0_63330473/article/details/135079898

def json_to_txt(jsonfilePath, resultDirPath, classList):
    
    jsonfileList = glob.glob(osp.join(jsonfilePath, "*.json"))

    for jsonfile in tqdm(jsonfileList, desc='Processing'):
        with open(jsonfile, "r", encoding='UTF-8') as f:
            file_in = json.load(f)
            
            # Read all annotation targets recorded in the file
            shapes = file_in["shapes"]
            
            # Create a txt file using the image name to save the data
            txt_file_path = osp.join(resultDirPath, osp.basename(jsonfile).replace(".json", ".txt"))
            with open(txt_file_path, "w") as file_handle:
                # Iterate over each target's contour in shapes
                for shape in shapes:
                    if shape["label"] not in classList:
                        classList.append(shape["label"])
                    # Find the ID of the category from classList according to the target's category label in the json and write it into the txt file
                    file_handle.writelines(str(classList.index(shape["label"])) + " ")
                    
                    # Initialize maximum and minimum values
                    x_max, y_max = -float('inf'), -float('inf')
                    x_min, y_min = float('inf'), float('inf')

                    # Iterate over each point in the shape contour to find the maximum and minimum x and y
                    for point in shape["points"]:
                        x = point[0] / file_in["imageWidth"]  # X-coordinate of a point in the mask contour
                        y = point[1] / file_in["imageHeight"]  # Y-coordinate of a point in the mask contour
                        if x > x_max:
                            x_max = x
                        if y > y_max:
                            y_max = y
                        if x < x_min:
                            x_min = x
                        if y < y_min:
                            y_min = y

                    # Write the four maximum and minimum points into the file
                    file_handle.writelines(f"{x_min} {y_min} {x_max} {y_max} ")
                    file_handle.writelines("\n")
            file_handle.close()
        f.close()
    with open(osp.join(resultDirPath, 'classes.txt'), 'w') as f:
        for i in classList:
            f.write(i + '\n')

if __name__ == "__main__":
    jsonfilePath = "./data_annotated"  # Directory containing the json files annotated with labelme, which are to be converted
    resultDirPath = "./output_txts"  # Directory where the generated txt files will be stored
    classList = []  # Initialize the class label list
    os.makedirs(resultDirPath, exist_ok=True)  # Ensure the output directory exists
    json_to_txt(jsonfilePath=jsonfilePath, resultDirPath=resultDirPath, classList=classList)
