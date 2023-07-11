#!/usr/bin/env python
'''
    Author: Devesh Sharma
    email: sharma.98@iitj.ac.in
'''
import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import h5py
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def model_name_to_type(mname):
    sp_name = mname.split('/')[-1].split('.')[0].split('_')
    return f'%s_%s'%(sp_name[1], sp_name[2])


def set_model_params(args):
    params = {'points_per_side': args.points_per_side,
              'pred_iou_thresh': args.pred_iou_thresh,
              'stability_score_thresh': args.stability_threshold,
              'crop_n_layers': args.crop_n_layers,
              'downscale_factor': args.downscale_factor,
              'min_mask_region_area': args.min_area}
    return params


def get_device():
    if torch.cuda.is_available is True:
        device = "cuda"
    else:
        device = "cpu"
    return device


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def visualize_output(im, masks):
    plt.figure(figsize=(50, 50))
    plt.imshow(im)
    show_anns(masks)
    plt.axis('off')
    plt.show()


def process_args():
    pars = argparse.ArgumentParser(description="Simple utility to perform segmentation"
                                   " on input images and generate .h5 database")
    pars.add_argument("-i", "--input", help="Input images to perform segmentation on",
                      type=str, required=True)
    pars.add_argument("-m", "--model", help="The model name to be used"
                      " copied into output", type=str, required=True)
    pars.add_argument("-o", "--output", type=str,
                      help="Output directory where the output database will be written",
                      required=False)
    pars.add_argument("-v", "--vizualize", help="Display the segmented output image",
                      action='store_true', required=False, default=False)
    pars.add_argument("-w", "--write-h5", help="Generates seg.h5 database file",
                      action='store_true', required=False, default=False)

    pars.add_argument("-p", "--points-per-side", help="Model parameter points per side",
                      required=False, default=32, type=int)
    pars.add_argument("-t", "--pred-iou-thresh", help="Model parameter prediction iou threshold",
                      required=False, default=0.88, type=float)
    pars.add_argument("-s", "--stability-threshold", help="Model parameter stability score threshold",
                      required=False, default=0.95, type=float)
    pars.add_argument("-c", "--crop-n-layers", help="Model parameter crop number layers",
                      required=False, default=0, type=int)
    pars.add_argument("-d", "--downscale-factor", help="Model parameter crop number points downscale factor",
                      required=False, default=1, type=int)
    pars.add_argument("-a", "--min-area", help="Model parameter minimum mask region area",
                      required=False, default=0, type=int)
    return pars.parse_args()


def sanity_check(args):
    if os.path.exists(args.input) is False:
        print(f'{args.input} does not exist.');
        return -1
    if os.path.exists(args.output) is False:
        print(f'{args.output} does not exist...created')
        os.makedirs(args.output)
    if os.path.exists(args.model) is False:
        print(f'{args.model} is not found.')
        return -1
    return 0


def load_model(args):
    model_type = model_name_to_type(args.model)
    sam = sam_model_registry[model_type](checkpoint=args.model)
    sam.to(device=get_device())
    params = set_model_params(args)
    return SamAutomaticMaskGenerator(sam, points_per_side = params['points_per_side'],
                                     pred_iou_thresh = params['pred_iou_thresh'],
                                     stability_score_thresh = params['stability_score_thresh'],
                                     crop_n_layers = params['crop_n_layers'],
                                     crop_n_points_downscale_factor = params['downscale_factor'],
                                     min_mask_region_area = params['min_mask_region_area']
                                     )


def init_h5_database(out_path):
    dbout = os.path.join(out_path, 'seg.h5')
    db = h5py.File(dbout, 'w')
    db.create_group('/data')
    return db


def add_to_h5_database(imname, masks, db):
    ''' masks.dict_keys(['segmentation', 'area', 'bbox',
                         'predicted_iou', 'point_coords',
                         'stability_score', 'crop_box',
                         'label'])
    '''
    for idx, msk in enumerate(masks):
        dname=f'%s_%d'%(imname, idx)
        db['/data'].create_dataset(dname, data=msk['segmentation'])
        db['/data'].attrs['area'] = msk['area']
        db['/data'].attrs['bbox'] = msk['bbox']
        db['/data'].attrs['label'] = msk['label']
        db['/data'].attrs['predicted_iou'] = msk['predicted_iou']
        db['/data'].attrs['point_coords'] = msk['point_coords']
        db['/data'].attrs['stability_score'] = msk['stability_score']
        db['/data'].attrs['crop_box'] = msk['crop_box']


def segment_one(im, segmentor, db, gen_h5, viz):
    imname = im.split('/')[-1].split('.')[0]

    image = cv2.imread(im)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = segmentor.generate(image)
    # Add label vector
    for idx, msk in enumerate(masks):
        msk['label'] = idx + 1

    if gen_h5:
        add_to_h5_database(imname, masks, db)

    if viz:
        visualize_output(image, masks)

    return masks


def segment_multiple(args, segmentor, db):
    for im in sorted(os.listdir(args.input)):
        impath = os.path.join(args.input, im)
        if os.path.isfile(impath) is False:
            continue
        _ = segment_one(impath, segmentor, db,
                        args.write_h5, args.vizualize)
    return None


def generate_masks(args, segmentor):
    db = None
    if args.write_h5:
        db = init_h5_database(args.output)

    if os.path.isdir(args.input) is False:
        masks = segment_one(args.input, segmentor, db,
                            args.write_h5, args.vizualize)
    else:
        masks = segment_multiple(args, segmentor, db)

    if args.write_h5:
        db.close()
    return masks


def start_main():
    args = process_args()
    if sanity_check(args):
        exit(-1)
    segmentor = load_model(args)
    masks = generate_masks(args, segmentor)


if __name__ == '__main__':
    start_main()
