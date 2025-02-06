import time
# from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from paddleocr import PaddleOCR, draw_ocr, PPStructure, draw_structure_result
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


class OCRHandler:
    def __init__(self):
        self.__ppeng = None
        # self._layout_engine = PPStructure(ocr=False, return_ocr_result_in_table=True, lang='ch')
        self._layout_engine = None
        self._table_engine = None
        # self.__table_engine = PPStructure(
        #     layout_model_dir=r'W:\future_exchange\modules\utils\picodet_lcnet_x1_0_fgd_layout_table_infer')
        self.__image_ocr_res_cache = {}
        self.__image_layout_cache = {}

    @staticmethod
    def preprocess_image(image_path, threshold=30):
        image = Image.open(image_path)
        image_array = np.array(image)
        upper_bound = np.array([255, 255, 255])
        lower_bound = np.array([255 - threshold, 255 - threshold, 255 - threshold])
        for i in range(3):
            mask = (image_array[:, :, i] >= lower_bound[i]) & (image_array[:, :, i] <= upper_bound[i])
            image_array[:, :, i][mask] = 255
        out = Image.fromarray(image_array)
        img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        return img

    def get_ocr_result(self, img_input, if_debug=True, out_image_dir=None, input_marker=None):
        raw_result = None
        if isinstance(img_input, str):
            if img_input in self.__image_ocr_res_cache:
                logger.warning(f"{img_input} already got ocr res in cache.")
                raw_result = self.__image_ocr_res_cache[img_input]
            img_input = Path(img_input)
            image = self.preprocess_image(image_path=img_input)
            if if_debug:
                out_image_dir = Path(img_input).parent if not out_image_dir else out_image_dir
                assert out_image_dir, 'You need to provide output image path'

        elif isinstance(img_input, Path):
            if str(img_input) in self.__image_ocr_res_cache:
                logger.warning(f"{img_input} already got ocr res in cache.")
                raw_result = self.__image_ocr_res_cache[str(img_input)]
            image = self.preprocess_image(image_path=img_input)
            if if_debug:
                out_image_dir = Path(img_input).parent if not out_image_dir else out_image_dir
                assert out_image_dir, 'You need to provide output image path'

        else:
            image = img_input
            if if_debug:
                assert out_image_dir, 'You need to provide output image path'
        if not raw_result and input_marker:
            raw_result = self.__image_ocr_res_cache.get(input_marker)
        if not raw_result:
            if not self.__ppeng:
                self.__ppeng = PaddleOCR(use_angle_cls=True, lang='en')
            try:
                raw_result = self.__ppeng.ocr(img=image, cls=False)
            except RuntimeError as e:
                self.__ppeng =  PaddleOCR(use_angle_cls=True, lang='en')
                raw_result = self.__ppeng.ocr(img=image, cls=False)

        if isinstance(img_input, Path) and str(img_input) not in self.__image_ocr_res_cache and raw_result:
            self.__image_ocr_res_cache[str(img_input)] = raw_result
        if input_marker not in self.__image_ocr_res_cache:
            self.__image_ocr_res_cache[input_marker] = raw_result

        if if_debug:
            if not raw_result[0]:
                return None
            boxes = [line[0] for line in raw_result[0]]
            txts = [line[1][0] for line in raw_result[0]]
            scores = [line[1][1] for line in raw_result[0]]
            im_show = draw_ocr(image, boxes, txts, scores, font_path=str(Path(__file__).parent / 'simfang.ttf'))
            im_show = Image.fromarray(im_show)
            out_image_path = out_image_dir / f'ocr_result_{str(int(time.time() * 1000))}.jpg'
            im_show.save(out_image_path)
        return raw_result

    @staticmethod
    def get_average_row_distance(ocr_result_raw):
        heights = []
        if not ocr_result_raw:
            return 0
        for r in ocr_result_raw[0]:
            _cur_row_height = (r[0][2][1] - r[0][0][1]) // 2 + r[0][0][1]
            # _cur_row_height = r[0][0][1]
            heights.append(_cur_row_height)
        diffs = []
        for i in range(len(heights) - 2):
            if heights[i + 1] - heights[i] >= 0:
                diffs.append(heights[i + 1] - heights[i])
        if 0 < len(diffs) <= 3:
            average_diff = sum(diffs) / len(diffs)
        elif len(diffs) == 0:
            average_diff = 0
        else:
            max_diffs = max(diffs)
            min_diffs = min(diffs)

            average_diff = (sum(diffs) - max_diffs - min_diffs) / (len(diffs) - 2)
        return average_diff

    @staticmethod
    def is_bbox_in_window(ocr_result_raw, bbox):
        for r in ocr_result_raw[0]:
            _cur_row_height = (r[0][2][1] - r[0][0][1]) // 2 + r[0][0][1]
            cur_center_x = (r[0][0][0] + r[0][2][0]) // 2
            cur_center_y = (r[0][0][1] + r[0][2][1]) // 2
            if bbox[0] < cur_center_x < bbox[2] and bbox[1] < cur_center_y < bbox[3]:
                return True

        return False

    def img_crop_method_0(self, img_input_path: Path, if_debug, output_path):
        image = Image.open(img_input_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return [image], ['text']

    def img_crop_method_1(self, img_input_path: Path, if_debug, output_path):
        image = Image.open(img_input_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        input_hash = str(img_input_path)
        _raw_result = self.get_ocr_result(img_input_path,
                                          input_marker=input_hash,
                                          if_debug=if_debug,
                                          out_image_dir=output_path)
        _threshold = self.get_average_row_distance(_raw_result)
        # 获取图片大小
        width, height = image.size
        # 计算上半部分和下半部分的高度
        half_height = height // 2
        # 切割图片
        top_half = image.crop((0, 0, width, half_height))
        bottom_half = image.crop((0, half_height - _threshold, width, height))
        return [top_half, bottom_half], ['text', 'text']

    def img_crop_method_2(self, img_input_path: Path, if_debug, output_path):
        input_hash = str(img_input_path)
        image = Image.open(img_input_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        width, height = image.size
        _raw_result = self.get_ocr_result(img_input_path,
                                          input_marker=input_hash,
                                          if_debug=if_debug,
                                          out_image_dir=output_path)
        _threshold = self.get_average_row_distance(_raw_result)

        raw_layout_res = self.layout_extraction(img_input_path,
                                                if_debug=if_debug,
                                                output_path=output_path)
        img_parts = []
        part_types = []
        part_details = []
        prev_y = 0
        prev_part_details = []
        for part in raw_layout_res:
            part_type = part.get('type')
            bbox = part.get('bbox')
            if bbox[1] > prev_y:
                is_bbox_in_region = self.is_bbox_in_window(_raw_result, (0, prev_y, width - 1, bbox[1]))
                if is_bbox_in_region:
                    cur_part_img = image.crop((0, prev_y, width - 1, bbox[1]))
                    img_parts.append(cur_part_img)
                    part_types.append('text')
                    part_details.append('')
            if 'caption' in part_type:
                prev_part_details.append((part_type.split('_')[0], bbox))
                continue
            if prev_part_details:
                _temp_stack = []
                _if_found_caption = False
                while prev_part_details:
                    _cur_part_detail = prev_part_details.pop()
                    if _cur_part_detail[0] == part_type:
                        cur_img_part = image.crop((min(_cur_part_detail[1][0], bbox[0]),
                                                   min(_cur_part_detail[1][1], bbox[1]),
                                                   max(_cur_part_detail[1][2], bbox[2]),
                                                   max(_cur_part_detail[1][3], bbox[3])))
                        img_parts.append(cur_img_part)
                        part_types.append(part_type)
                        _if_found_caption = True
                        break
                    _temp_stack.append(_cur_part_detail)
                while _temp_stack:
                    b_e = _temp_stack.pop()
                    prev_part_details.append(b_e)
                if not _if_found_caption:
                    cur_img_part = image.crop(bbox)
                    img_parts.append(cur_img_part)
                    part_types.append(part_type)
            else:
                cur_img_part = image.crop(bbox)
                img_parts.append(cur_img_part)
                part_types.append(part_type)
        return img_parts, part_types

    def img_crop_method_3(self, img_input_path: Path, if_debug, output_path):
        # input_hash = str(img_input_path)
        image = Image.open(img_input_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        width, height = image.size
        raw_layout_res = self.layout_extraction(img_input_path,
                                                if_debug=if_debug,
                                                output_path=output_path)
        potential_images = [img_input_path]
        if width > height:
            for middle_x in range(width // 2, width):
                if_cut_in_middle = True
                for layout in raw_layout_res:
                    if layout['bbox'][0] <= middle_x <= layout['bbox'][2]:
                        logger.warning("Cannot cut in middle.")
                        if_cut_in_middle = False
                        break
                if if_cut_in_middle:
                    logger.info(f"Cut at x={middle_x}")
                    # 切割图片
                    left_half = image.crop((0, 0, middle_x, height))
                    right_half = image.crop((middle_x, 0, width, height))
                    potential_out_parts = [left_half, right_half]
                    potential_images = []
                    for idx, part in enumerate(potential_out_parts):
                        part_img_path = f'{img_input_path.parent / (str(f"{img_input_path.stem}") + str(f"_{idx}.png"))}'
                        part.save(part_img_path)
                        potential_images.append(part_img_path)
                    break
        # Structure parts
        out_img_parts = []
        out_type = []
        for img_part_path in potential_images:
            layout_raw = self.layout_extraction(img_input_path=img_part_path, if_debug=if_debug)
            c_image = Image.open(img_part_path)
            if c_image.mode == "RGBA":
                c_image = c_image.convert("RGB")
            for layout in layout_raw:
                cur_img_part = c_image.crop((layout['bbox'][0],
                                             layout['bbox'][1],
                                             layout['bbox'][2],
                                             layout['bbox'][3]))
                out_img_parts.append(cur_img_part)
                out_type.append(layout['type'])
        return out_img_parts, out_type

    def get_ocr_result_by_row(self, img_input_path, threshold=None, if_debug=True, with_line_source_box=False,
                              do_not_format=False, crop_method=0, output_path=None):
        lines = []
        line_bounding_boxes = []

        if not img_input_path:
            logger.warning(f"Input img_input_path is {img_input_path}")
            if with_line_source_box:
                return lines, line_bounding_boxes
            return lines

        if isinstance(img_input_path, str):
            img_input_path = Path(img_input_path)
        if not output_path:
            output_path = img_input_path.parent

        image_hash = str(img_input_path)

        if crop_method == 0:
            to_parse, part_types = self.img_crop_method_0(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 1:  # HALF CROP
            to_parse, part_types = self.img_crop_method_1(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 2:
            to_parse, part_types = self.img_crop_method_2(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 3:
            to_parse, part_types = self.img_crop_method_3(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        else:
            logger.warning(f"Crop method: {crop_method} is not supported. Will use default method 0")
            to_parse, part_types = self.img_crop_method_0(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        for part_idx, img_input in enumerate(to_parse):
            img_input = np.array(img_input)
            # 保存切割后的图片
            raw_result = self.get_ocr_result(img_input,
                                             if_debug=if_debug,
                                             out_image_dir=output_path,
                                             input_marker=image_hash + str(part_idx))
            _threshold = self.get_average_row_distance(raw_result) if not threshold else threshold
            logger.warning(f"Threshold will use averaged value:{_threshold}")
            _cur_row_height = 0
            _prev_cell = None
            _cur_row_content = []
            _cur_bounding_box = []
            for r in raw_result[0]:

                if abs((r[0][1][1] - r[0][0][1]) / (r[0][1][0] - r[0][0][0])) >= 0.3:
                    logger.warning(r)
                    logger.warning(f"SLOPE: {(r[0][1][1] - r[0][0][1]) / (r[0][1][0] - r[0][0][0])}")
                    continue
                if (r[0][2][1] - r[0][0][1]) // 2 + r[0][0][1] - _cur_row_height >= _threshold - 10:
                    # if r[0][0][1] - _cur_row_height >= threshold:
                    if _cur_row_content:
                        if with_line_source_box:
                            lines.append(_cur_row_content)
                        else:
                            lines.append(' '.join(_cur_row_content))
                        line_bounding_boxes.append(_cur_bounding_box)
                        _cur_row_content = []
                        _cur_bounding_box = []
                        _prev_cell = None
                    if not do_not_format:
                        if _prev_cell:
                            gap_num = ((r[0][2][0] - r[0][0][0]) // 2 + r[0][0][0] - (
                                    (_prev_cell[0][2][0] - _prev_cell[0][0][0]) // 2 + _prev_cell[0][0][
                                0])) // _threshold
                            gap_num = int(gap_num)
                            if gap_num >= 2:
                                _cur_row_content.append(int(gap_num // 2) * " ")
                                _cur_row_content.append('|')
                                _cur_row_content.append(int(gap_num // 2) * " ")
                        else:
                            gap_num = ((r[0][2][0] - r[0][0][0]) // 2 + r[0][0][0] - 0) // _threshold
                            _cur_row_content.append(int(gap_num) * " ")
                _cur_row_content.append(r[1][0])
                _cur_bounding_box.append(r[0])
                _cur_row_height = (r[0][2][1] - r[0][0][1]) // 2 + r[0][0][1]
                _prev_cell = r
            if _cur_row_content:

                if with_line_source_box:
                    lines.append(_cur_row_content)
                else:
                    lines.append(' '.join(_cur_row_content))
                line_bounding_boxes.append(_cur_bounding_box)
                _cur_row_content = []
                _cur_bounding_box = []
        if with_line_source_box:
            return lines, line_bounding_boxes
        return lines

    def get_ocr_result_by_block(self, img_input_path, if_debug=True, with_line_source_box=False, crop_method=0,
                                output_path=None):
        lines = []
        line_bounding_boxes = []

        if not img_input_path:
            logger.warning(f"Input img_input_path is {img_input_path}")
            if with_line_source_box:
                return lines, line_bounding_boxes
            return lines

        if isinstance(img_input_path, str):
            img_input_path = Path(img_input_path)
        if not output_path:
            output_path = img_input_path.parent

        image_hash = str(img_input_path)

        if crop_method == 0:
            to_parse, part_types = self.img_crop_method_0(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 1:  # HALF CROP
            to_parse, part_types = self.img_crop_method_1(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 2:
            to_parse, part_types = self.img_crop_method_2(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        elif crop_method == 3:
            to_parse, part_types = self.img_crop_method_3(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        else:
            logger.warning(f"Crop method: {crop_method} is not supported. Will use default method 0")
            to_parse, part_types = self.img_crop_method_0(img_input_path,
                                                          if_debug=if_debug,
                                                          output_path=output_path
                                                          )
        output = []
        for part_idx, img_input in enumerate(to_parse):
            part_img_path = f'{img_input_path.parent / (str(f"{img_input_path.stem}") + str(f"_part_{part_idx}.png"))}'
            img_input.save(part_img_path)
            img_input = np.array(img_input)
            # 保存切割后的图片
            if part_types[part_idx] in ['title', 'footer', 'header', 'text']:
                raw_result = self.get_ocr_result(img_input,
                                                 if_debug=if_debug,
                                                 out_image_dir=output_path,
                                                 input_marker=image_hash + str(part_idx))
                if not raw_result:
                    continue
                output.extend([i[1][0] for i in raw_result[0]])
            else:
                if not self._table_engine:
                    self._table_engine = PPStructure(layout=False, lang='en')
                try:
                    structured_result = self._table_engine(img_input)
                except RuntimeError as e:
                    self._table_engine = PPStructure(layout=False, lang='en')
                    structured_result = self._table_engine(img_input)

                # for res in sorted_layout_boxes(structured_result, img_input.shape[1]):
                for res in structured_result:
                    _result = res['res']
                    if _result.get('html'):
                        output.append(_result.get('html'))
                    else:
                        raw_result = self.get_ocr_result(img_input,
                                                         if_debug=if_debug,
                                                         out_image_dir=output_path,
                                                         input_marker=image_hash + str(part_idx))
                        if not raw_result:
                            continue
                        output.append([i[1][0] for i in raw_result[0]])
        return output

    def layout_extraction(self, img_input_path, if_debug=True, output_path=None):
        if isinstance(img_input_path, str):
            img_input_path = Path(img_input_path)
        if not output_path:
            output_path = img_input_path.parent
        if not self._layout_engine:
            self._layout_engine = PPStructure(ocr=False, return_ocr_result_in_table=True, lang='en')
        try:
            layout_res_raw = self._layout_engine(str(img_input_path))
        except RuntimeError as e:
            self._layout_engine = PPStructure(ocr=False, return_ocr_result_in_table=True, lang='en')
            layout_res_raw = self._layout_engine(str(img_input_path))

        if if_debug:
            font_path = str(Path(__file__).parent / 'simfang.ttf')
            image = Image.open(img_input_path).convert('RGB')
            im_show = draw_structure_result(image, layout_res_raw, font_path=font_path)
            im_show = Image.fromarray(im_show)
            im_show.save(output_path / f'ocr_layout_{img_input_path.stem}_{str(int(time.time() * 1000))}.jpg')
        for i in layout_res_raw:
            i.pop('img')
        return layout_res_raw


if __name__ == "__main__":
    img_path = Path(r"/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/pdfpro/storage/dior-receipt-template.png")
    inst = OCRHandler()
    # print(inst.get_ocr_result(img_path))
    # layouts = inst.layout_extraction(img_path)
    outs = inst.get_ocr_result_by_block(img_path, if_debug=True)
    print(outs)
