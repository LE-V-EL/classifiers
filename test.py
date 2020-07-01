import os

import argparse

import perception.classifier as c

def main():
    parser = argparse.ArgumentParser(description="Training the segmentation pipeline")
    parser.add_argument("--path", nargs='?', help="path to storage folder with dataset folder in it, defaults to current folder", 
        type=str, default=os.path.dirname(__file__))

    parser.add_argument("--model_name", help="maskrcnn training epochs", type=str)
    parser.add_argument("--gpu", nargs='?', help="select the gpu", type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    classifier = c.Classifier(storage_dir=args.path)

    classifier.test(args.model_name)


if __name__ == '__main__':
    main()
