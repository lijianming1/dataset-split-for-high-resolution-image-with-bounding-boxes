from lib import TransformVOCDataset

# data_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC'
data_url = r'/windata/f/computer_vision/221213-袜子外观检测/袜子原图-汇总/all_voc'
# data_url = r'test/data_voc'
# STEP: [execute transform, show transform]
STEP = [1, 1]
tfd = TransformVOCDataset(data_url)
if STEP[0]:
    # begin to transform dataset
    tfd.transform()
if STEP[1]:
    # view transformed dataset
    tfd.show_crop_with_annotation('/windata/f/computer_vision/221213-袜子外观检测/袜子原图-汇总/all_voc_trans/Annotations')