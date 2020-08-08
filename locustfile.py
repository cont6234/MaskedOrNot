from locust import HttpUser, task, constant, TaskSet
import os
import random


class APITasks(TaskSet):
    wait_time = constant(0)
    # images dir to choose from
    images_dirpath = os.environ['IM_DIR']
    impaths = None
    random_impath = None
    random_image = None

    def get_im_paths(self):
        path_list = []
        for path, subdirs, files in os.walk(self.images_dirpath):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
                    # if name.endswith(".jpeg"):
                    get_path = os.path.join(path, name)
                    path_list.append(get_path)
        return path_list

    def on_start(self):
        self.impaths = self.get_im_paths()

    # providing 1 as parameteres to task() ensures that both the tasks will be performed at 1:1 ratio
    @task(1)
    def detect_mask_api(self):
        random_index = random.randrange(len(self.impaths))
        # select a randome image from the list of images available in the IM_DIR
        self.random_impath = self.impaths[random_index]
        self.random_image = self.get_imdata()

        data = self.random_image
        headers = {
            'Content-Type': 'image/jpeg'
        }
        url = '/detect_mask'
        response = self.client.post(url=url, headers=headers,
                                    data=data)

        print(response)

    @task(1)
    def detect_face_api(self):
        random_index = random.randrange(len(self.impaths))
        self.random_impath = self.impaths[random_index]
        self.random_image = self.get_imdata()

        data = self.random_image
        headers = {
            'Content-Type': 'image/jpeg'
        }
        url = '/detect_faces?rank=1'
        response = self.client.post(url=url, headers=headers,
                                    data=data)

        print(response.json())

    def get_imdata(self):
        with open(self.random_impath, "rb") as f:
            data = f.read()
        return data


class TestAPITasks(HttpUser):
    tasks = [APITasks]
