import numpy as np
import xml.etree.ElementTree as ET

class XMLParser(object):
    @classmethod
    def get_bbox_info(cls, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        objects = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = [int(np.floor(eval(x.text))) for x in obj.find('bndbox')]
            """
                -----------------
                in case of abnormal image width and height, rotation 90 degree and shift image, 
                in order to make top left point as zero point.
                -----------------
                transform rectangle label boxes to the new image coordinate system. transform function 
                located in lib/rotation_shift.py
                -----------------
                add by jmingl@tju.edu.cn at 2023.5.5
            """
            if width < height:
                from rotation_shift import transform_points
                # shape: [2, 2]
                q_points = transform_points([bbox[:2], bbox[2:]], width, height, 90)
                bbox = q_points.flatten().astype(np.int32).tolist()
                width, height = height, width
            objects.append({'label': label, 'bbox': bbox})

        xml_info = {}
        (lambda d, **kwargs: d.update(**kwargs)) (xml_info, filename=filename, size={'width':width, 'height':height}, objects=objects)

        return xml_info

    @classmethod
    def gen_xml_file(cls, bbox_info, xml_file_save='./test_out.xml'):
        """

        :param bbox_info: {
                                filename: xxx,
                                size: {
                                            height: xxx,
                                            width: xxx,
                                            depth: xxx
                                        }
                                objects:[
                                            {
                                                name: xxx
                                                bndbox:{
                                                    xmin: xxx,
                                                    ymin: xxx,
                                                    xmax: xxx,
                                                    ymax: xxx
                                                },

                                            },
                                           {
                                                name: xxx
                                                bndbox:{
                                                    xmin: xxx,
                                                    ymin: xxx,
                                                    xmax: xxx,
                                                    ymax: xxx
                                                },

                                            },,...
                                       ]
                            }
        :return:
        """

        def create_xml_tree(tree_depth=0, node=None, **kwargs):
            # for only one node
            if len(kwargs) == 1:
                # root node or sub node
                if tree_depth == 0:
                    # for root node
                    node = ET.Element(list(kwargs.keys())[0])
                    tree_depth += 1
                    # just jump in to next level
                    create_xml_tree(tree_depth, node=node, **dict(list(kwargs.values())[0]))
                    ET.dump(node)
                    tree = ET.ElementTree(node)
                    return tree
                else:
                    # for sub-level with one node
                    if list(kwargs.keys())[0] == 'objects':
                        sub_node = node
                    else:
                        sub_node = ET.SubElement(node, list(kwargs.keys())[0])


                if not isinstance(list(kwargs.values())[0], str|int|list) and len(list(kwargs.values())[0]) > 1:
                    # for sub level has multiple nodes
                    tree_depth += 1
                    # jump into level with multiple nodes
                    create_xml_tree(tree_depth, sub_node, **dict(list(kwargs.values())[0]))
                else:
                    if not isinstance(list(kwargs.values())[0], list|dict):
                        sub_node.text = str(list(kwargs.values())[0])

                    else:
                        for obj in list(kwargs.values())[0]:
                            create_xml_tree(tree_depth, sub_node, object=obj)


            elif len(kwargs) > 1:
                # multiple nodes
                # jump into next tree layer
                tree_depth += 1
                for k, v in kwargs.items():
                    create_xml_tree(tree_depth, node, **{k:v})
                    ET.dump(node)


        tree = create_xml_tree(annotation=bbox_info)

        with open(xml_file_save, 'wb') as f:
            tree.write(f)

    @classmethod
    def xml_bbox_info_to_array(cls, xml_info):
        bbox_info = xml_info.get('objects')
        labels = []
        bboxes = []

        for box in bbox_info:
            labels.append(box.get('label'))
            xmin, ymin, xmax, ymax = box.get('bbox')
            bboxes.append([xmin, ymin, xmax, ymax])

        return np.array(labels), np.array(bboxes)

