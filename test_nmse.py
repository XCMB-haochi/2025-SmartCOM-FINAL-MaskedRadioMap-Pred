#!/usr/bin/env python3
"""
Calculate NMSE for SPARTA-Net predictions
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths
PRED_PATH = 'output_unet'
GT_PATH = 'Data/OriginDataSet'

# Test base stations
TEST_BS = [7, 12, 25, 30, 38, 45, 52, 60, 65, 74]

def calculate_nmse():
  nmse_list = []

  # Get all prediction files
  pred_files = [f for f in os.listdir(PRED_PATH) if f.startswith('pred_') and f.endswith('.png')]

  for pred_file in tqdm(pred_files, desc='Calculating NMSE'):
      # Parse filename: pred_{map_id}_{bs_id}.png
      parts = pred_file[5:-4].split('_')  # Remove 'pred_' and '.png'
      map_id = parts[0]
      bs_id = parts[1]

      # Load prediction
      pred = np.array(Image.open(os.path.join(PRED_PATH, pred_file)).convert('L'), dtype=np.float32) / 255.0

      # Load ground truth
      gt_file = f'{map_id}_{bs_id}.png'
      gt = np.array(Image.open(os.path.join(GT_PATH, gt_file)).convert('L'), dtype=np.float32) / 255.0

      # Calculate NMSE for this image
      mse = np.mean((pred - gt) ** 2)
      nmse_list.append(mse)

  # Average NMSE
  avg_nmse = np.mean(nmse_list)

  print(f"Average NMSE: {avg_nmse:.4e}")

  return avg_nmse

if __name__ == '__main__':
  calculate_nmse()
