from detectron2.data.datasets import register_coco_instances

def load_datasets():
    voc_image_path = 'voc2007/VOCdevkit/VOC2007/JPEGImages'
    coco_image_path = '/home/work/yujin/data/train2014/'
    coco_val_image_path = '/home/work/yujin/data/coco2017/val2017'

    pascal_voc_2007_trainval = "./data_jsons/voc_trainval_2007.json"
    register_coco_instances("custom_pascal_voc_trainval_2007", {},
                            pascal_voc_2007_trainval, voc_image_path)

    pascal_voc_2007_test = "./data_jsons/voc_test_2007.json"
    register_coco_instances("custom_pascal_voc_test_2007", {},
                            pascal_voc_2007_test, voc_image_path)

    coco_2014_train = './data_jsons/instances_train2014_no_annotations.json'
    register_coco_instances("custom_coco2014_train", {},
                            coco_2014_train, coco_image_path)
    coco_minival = './data_jsons/instances_coco_minival.json'
    register_coco_instances("custom_coco_minival", {},
                            coco_minival, coco_val_image_path)

