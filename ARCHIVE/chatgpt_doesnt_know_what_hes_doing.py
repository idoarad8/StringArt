import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.util import img_as_float
from skimage.draw import line_aa
from skimage.transform import radon
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

# Step 1: Load a grayscale patch from a real image
image_rgb = data.coffee()
image_gray = img_as_float(color.rgb2gray(image_rgb))

patch_size = 256
start_y = image_gray.shape[0] // 2 - patch_size // 2
start_x = image_gray.shape[1] // 2 - patch_size // 2
patch = image_gray[start_y:start_y+patch_size, start_x:start_x+patch_size]

# Step 2: Draw a grayscale line segment using anti-aliasing
p1 = (30, 200)    # (row, col)
p2 = (0, 0)

line_only = np.zeros_like(patch)
rr, cc, val = line_aa(*p1, *p2)
line_only[rr, cc] = val  # insert anti-aliased line into blank image

# Step 3: Compute true Radon transform of the line-only image
theta = np.linspace(0., 180., max(patch.shape), endpoint=False)
sinogram_true = radon(line_only, theta=theta)

# Step 4: Approximate Radon transform of the same line segment
def radon_line_segment(image, p1, p2, angles, num_samples=500, r_bins=256):
    """
    Approximate Radon transform of a line segment using skimage-compatible geometry.
    Returns sinogram (r_bins x len(angles)) and r-axis values.
    """
    y_vals = np.linspace(p1[0], p2[0], num_samples)
    x_vals = np.linspace(p1[1], p2[1], num_samples)
   
    coords = np.vstack((y_vals, x_vals))
    intensities = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)

    cy, cx = np.array(image.shape) / 2
    x_vals -= cx
    y_vals -= cy

    r_max = np.hypot(*image.shape) / 2
    r_edges = np.linspace(-r_max, r_max, r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    sinogram = np.zeros((r_bins, len(angles)))

    for i, theta_deg in enumerate(angles):
        theta_rad = np.deg2rad(theta_deg)
        # Use skimage-compatible projection equation
        r = x_vals * np.sin(theta_rad) - y_vals * np.cos(theta_rad)
        hist, _ = np.histogram(r, bins=r_edges, weights=intensities)
        sinogram[:, i] = hist

    return sinogram, r_centers

# Compute the approximation
sinogram_approx, r_vals = radon_line_segment(line_only, p1, p2, theta, num_samples=1000)

# Step 5: Resample the approximate sinogram to match r-axis of skimage.radon
r_true = np.linspace(-np.hypot(*line_only.shape)/2, np.hypot(*line_only.shape)/2, sinogram_true.shape[0])
sinogram_approx_resampled = np.zeros_like(sinogram_true)

for i in range(len(theta)):
    f = interp1d(r_vals, sinogram_approx[:, i], bounds_error=False, fill_value=0.0)
    sinogram_approx_resampled[:, i] = f(r_true)

# Step 6: Plot and compare
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].imshow(sinogram_true, extent=(0, 180, r_true[0], r_true[-1]), aspect='auto', cmap='gray')
axs[0].set_title("True Radon Transform (skimage)")
axs[0].set_xlabel("Angle (degrees)")
axs[0].set_ylabel("r (pixels)")

axs[1].imshow(sinogram_approx_resampled, extent=(0, 180, r_true[0], r_true[-1]), aspect='auto', cmap='gray')
axs[1].set_title("Approximate Radon Transform (Segment Only)")
axs[1].set_xlabel("Angle (degrees)")
axs[1].set_ylabel("r (pixels)")

diff = np.abs(sinogram_true - sinogram_approx_resampled)
im = axs[2].imshow(diff, extent=(0, 180, r_true[0], r_true[-1]), aspect='auto', cmap='hot')
axs[2].set_title("Absolute Difference")
axs[2].set_xlabel("Angle (degrees)")
axs[2].set_ylabel("r (pixels)")
fig.colorbar(im, ax=axs[2], shrink=0.7)

plt.tight_layout()
plt.show()