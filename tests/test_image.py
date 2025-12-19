"""Test the creation of a mesh from an image."""

from tifffile import imread

from vertax.pbc import PBCMesh

img = imread("tests/test_image.tif")  # [:-101, :]  # non rect, odd and pair dimensions


# mesh = PBCMesh.periodic_from_image(img)
# mesh.save_mesh("image_mesh.npz")

mesh = PBCMesh.load_mesh("image_mesh.npz")
mesh.plot()
