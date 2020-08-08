import argparse
import os
import requests
from sklearn.metrics import classification_report

rs = requests.session()


# returns lists of filenames and abs filepaths of a directory
def get_filenames_and_full_paths_for_images(base_dir):
    path_list = []
    image_names = []
    for path, subdirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png") or name.endswith(".txt"):
                # if name.endswith(".jpeg"):
                get_path = os.path.join(path, name)
                path_list.append(get_path)
                image_names.append(name)
    return image_names, path_list


# returns response of detect_mask api
def call_face_mask_detection_api(filepath):
    # data = cv2.imread(filepath)
    url = 'http://{}:{}/detect_mask'.format(args.ip, args.port)
    with open(filepath, "rb") as f:
        data = f.read()
    headers = {
        'Content-Type': 'image/jpeg'
    }

    resp = rs.post(url=url, headers=headers,
                   data=data)

    resp = resp.json()
    print(resp)

    return resp


# prepare predictions based on detect_mask responses; if masked then 1 otherwise 0 is appended in the list of predictions
def get_preds(filepaths):
    preds = []
    for i in range(0, len(filepaths)):
        fp = filepaths[i]
        resp = call_face_mask_detection_api(fp)

        if resp['mask'] > resp['no-mask']:
            preds.append(1)
        else:
            preds.append(0)
    return preds


# check if the path provided as argument is a directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification report for a directory containing all masked images')
    parser.add_argument('--dirpath', default='/home/tigerit/PycharmProjects/MaskedOrNot/dataset/images_all_final',
                        type=dir_path,
                        help='directory containing all masked images')
    parser.add_argument('--ip', default='localhost', type=str, help='mask detection ip')
    parser.add_argument('--port', default='5000', type=str, help='mask detection port')
    args = parser.parse_args()

    fnames, fpaths = get_filenames_and_full_paths_for_images(args.dirpath)

    y_trues = [1] * len(fnames)
    y_preds = get_preds(fpaths)

    # classification_report(y_trues, y_preds)
    target_names = ['no-mask', 'mask']

    result = classification_report(y_trues, y_preds, target_names=target_names, output_dict=True, zero_division=0)
    # as gts contain only masked photos, we will focus on only the results of class 'mask'
    print("For class 'mask' : ")
    print(result['mask'])
