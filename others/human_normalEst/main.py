import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import argparse
from pathlib import Path

from module.dataProcess import ImgSet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputDir",
        type=str,
        default='input',
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        default='normal',
    )
    # print(parser.parse_args())
    return parser.parse_args()


def main():
    args=get_args()

    myImgSet=ImgSet(Path(args.inputDir), Path(args.outputDir))

    model_id = 'Damo_XR_Lab/cv_human_monocular-normal-estimation'
    estimator = pipeline(Tasks.human_normal_estimation, model=model_id)

    for inPath, outPath  in myImgSet:
        result = estimator(str(inPath))
        normals_vis = result[OutputKeys.NORMALS_COLOR]
        cv2.imwrite(str(outPath), normals_vis)

if __name__ == '__main__':
    main()