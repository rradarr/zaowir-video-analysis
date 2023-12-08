import cv2
import cv2.typing
from pathlib import Path

class Image:
    base_dir_path = Path.cwd() / "s3"

    file_path: Path
    rgb_data: cv2.typing.MatLike
    bw_data: cv2.typing.MatLike

    def __init__(self, filename: str, dont_load: bool = False):
        self.file_path = Path(filename)
        if not dont_load:
            self.load_img()

    def load_img(self):
        self.rgb_data = cv2.imread(str(self.file_path))
        self.bw_data = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2GRAY)

    def show(self):
        cv2.imshow(self.file_path.name, self.rgb_data)
        cv2.waitKey()