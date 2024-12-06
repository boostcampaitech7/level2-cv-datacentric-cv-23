import os
import json
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Rainbow bbox image with id
def save_bbox_images(data_dir, save_dir, lang_list, split='train'):

    # id 글씨가 잘 보이도록 밝은 노란색과 연두색을 제외하고 새로운 cmap을 생성
    color_list = plt.cm.gist_rainbow(np.linspace(0, 0.15, 50).tolist() + np.linspace(0.3, 1, 150).tolist())
    new_cmap = LinearSegmentedColormap.from_list("modified_gist_rainbow", color_list)

    for lang in lang_list:
        anno_path = os.path.join(data_dir, '{}_receipt/ufo/{}.json'.format(lang, split))
        save_path = os.path.join(save_dir, lang)
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)

        with open(anno_path, 'r') as f:
            anno_data = json.load(f)

        for img_name, img_data in list(anno_data['images'].items()):
            img_path = os.path.join(data_dir, '{}_receipt/img/train/{}'.format(lang, img_name))

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert('RGB') # EXIF 회전 정보를 반영하여 이미지 회전

            plt.figure(figsize=(15, 18))
            plt.imshow(img)

            words = img_data['words']
            colors = new_cmap(np.linspace(0, 1, len(words)))

            for (word_id, word_info), color in zip(words.items(), colors):
                points = np.array(word_info['points'])
                points = np.append(points, [points[0]], axis=0)
                plt.plot(points[:,0], points[:,1], '-', color=color, linewidth=1, alpha=0.7)
                
                # id text 추가
                x_min = min(points[:, 0])
                y_min = min(points[:, 1])
                plt.text(x_min, y_min - 7,
                        f'{word_id}',
                        color=color,
                        fontsize=7,
                        ha='left',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2))

            plt.axis('off')
            plt.savefig(os.path.join(save_path, img_name), bbox_inches='tight', pad_inches=0)
            plt.close()

save_bbox_images('/data/ephemeral/home/code/data', '/data/ephemeral/home/code/test', lang_list=['sroie'], split='sroie_bbox')