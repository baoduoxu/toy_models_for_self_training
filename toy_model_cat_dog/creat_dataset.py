import os
import shutil
import random


def create_sample_dataset(source_dir_cat, source_dir_dog, sample_size=50, output_dir='./sample_data'):
    """
    从指定的猫和狗文件夹中分别抽取样本，并保存到指定的目标文件夹中。

    参数:
    - source_dir_cat: 包含猫图片的文件夹路径
    - source_dir_dog: 包含狗图片的文件夹路径
    - sample_size: 从每个类别中抽取的样本数量
    - output_dir: 保存样本的目标文件夹路径

    返回:
    - sample_dir_cat: 抽取的猫样本保存路径
    - sample_dir_dog: 抽取的狗样本保存路径
    """
    # 定义目标路径
    sample_dir_cat = os.path.join(output_dir, 'data/Cat')
    sample_dir_dog = os.path.join(output_dir, 'data/Dog')

    # 创建保存样本的目录
    os.makedirs(sample_dir_cat, exist_ok=True)
    os.makedirs(sample_dir_dog, exist_ok=True)

    # 从猫文件夹中抽取指定数量的图片
    cat_images = os.listdir(source_dir_cat)
    cat_sample_images = random.sample(cat_images, sample_size)
    for image in cat_sample_images:
        src = os.path.join(source_dir_cat, image)
        dst = os.path.join(sample_dir_cat, image)
        shutil.copy(src, dst)

    # 从狗文件夹中抽取指定数量的图片
    dog_images = os.listdir(source_dir_dog)
    dog_sample_images = random.sample(dog_images, sample_size)
    for image in dog_sample_images:
        src = os.path.join(source_dir_dog, image)
        dst = os.path.join(sample_dir_dog, image)
        shutil.copy(src, dst)

    print(f"{sample_size} 张猫的样本和 {sample_size} 张狗的样本已保存到 {output_dir}")

    return sample_dir_cat, sample_dir_dog


# 示例使用
source_dir_cat = r'data\\cat'
source_dir_dog = r'data\\dog'
sample_dir_cat, sample_dir_dog = create_sample_dataset(source_dir_cat, source_dir_dog, sample_size=1500,
                                                       output_dir='sample_data')
