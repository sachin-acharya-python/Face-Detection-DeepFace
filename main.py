from deepface import DeepFace, models
from tkinter import filedialog, Tk
from pandas import DataFrame
import os
import shutil

o = {
    '1.jpg': './output/Folder-1',
    '2.jpg': './output/Folder-2'
}

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "Human-beings"
]


def get_files():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    return filedialog.askopenfilenames(filetypes=[
        ("All Images", ('*.jpg', '*.jpeg', '*.png', '*.webp')),
        ('JPEG', ('*.jpg', '*.jpeg')),
        ('PNG', '*.png'),
        ('WEBP', '*.webp')
    ])


targets: list[str] = get_files()
try:
    os.remove('./references/*.pkl')
except:
    pass
for target in targets:
    try:
        dfs: list[DataFrame] = DeepFace.find(
                target, './references',
                silent=True,
                model_name=models[0], # 4
            )
        output = str(dfs[0]['identity'][0])
        output_path = o[os.path.basename(output)]
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        shutil.move(target, output_path)
    except Exception as e:
        print(e, '\t\n\n', target)
