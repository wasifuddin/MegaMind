import argparse
import logging
import os

import cv2
import torch
from mivolo.data.data_reader import get_all_files
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
import streamlit as st
_logger = logging.getLogger("inference")


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    return parser
@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    st.title('Megamind')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title('Face Mesh Application using MediaPipe')
    st.sidebar.subheader('Parameters')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)

    st.markdown(' ## Output')
    stframe = st.empty()

    DEMO_VIDEO = "/content/drive/MyDrive/res_data/h.mp4"
    # image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]
    cap = cv2.VideoCapture(DEMO_VIDEO)
    i = 0
    out = cv2.VideoWriter("/content/drive/MyDrive/res_data/v3.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    st.sidebar.text('Input Video')
    st.sidebar.video(DEMO_VIDEO)
    while True:

        ret, img = cap.read()
        i = i + 1
        if ret:
            # if i % 100 == 0:
            if True:

                detected_objects, out_im = predictor.recognize(img)

                if args.draw:
                    # bname = os.path.splitext(os.path.basename(img_p))[0]
                    filename = os.path.join(args.output, f"out_{i}.jpg")
                    cv2.imwrite(filename, out_im)
                    print("Output ", out_im.shape)
                    _logger.info(f"Saved result to {filename}")
                    # out_im = cv2.resize(out_im, (640,640))
                    out.write(out_im)
                    out_im = image_resize(image=out_im, width=640)

                    stframe.image(out_im, channels='RGB', use_column_width=True)
        else:
            out.release()
            break


if __name__ == "__main__":
    main()