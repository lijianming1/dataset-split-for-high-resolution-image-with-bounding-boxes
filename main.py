from lib import TransformVOCDataset

data_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC'

tfd = TransformVOCDataset(data_url)
tfd.transform()