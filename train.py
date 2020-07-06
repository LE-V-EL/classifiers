import os

import argparse

import perception.classifier as c

def main():
    parser = argparse.ArgumentParser(description="Training the segmentation pipeline")
    parser.add_argument("--path", nargs='?', help="path to storage folder with dataset folder in it, defaults to current folder", 
        type=str, default=os.path.dirname(__file__))

    parser.add_argument("--m_epoch", nargs='?', help="maskrcnn training epochs", type=int, default=1)
    parser.add_argument("--r_epoch", nargs='?', help="regression training epochs", type=int, default=1)
    parser.add_argument("--regres_net", nargs='?', help="regression network architecture", type=str, default="VGG19")

    parser.add_argument("--gpu", nargs='?', help="select the gpu", type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    classifier = c.Classifier(storage_dir=args.path, regres_net=args.regres_net)

    classifier.train(maskrcnn_epochs=args.m_epoch, vgg19_epochs=args.r_epoch)

    classifier.test()


if __name__ == '__main__':
    main()
