# Ensure the target folder path exists, or the program has permission to create it
# If the dataset is large, downloading may take some time
# If the download is interrupted, you can rerun the same code and it will automatically resume from the breakpoint

# 2. Non-edge cell dataset shape-matched to edge cells: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_nonedge_cell_image_dataset
# 3. Colony edge cell dataset: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_edge_cell_image_dataset
# 4. Interphase (i1) cell dataset shape-matched to prophase (m1) cells: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_i1_cell_image_dataset
# 5. Prophase (m1) cell dataset: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_m1_cell_image_dataset
# 6. Interphase (i2) cell dataset shape-matched to early prometaphase (m2) cells: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_i2_cell_image_dataset
# 7. Early prometaphase (m2) cell dataset: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_m2_cell_image_dataset
import quilt3 as q3

# Use raw strings to represent paths
q3.Package.install(
    "aics/hipsc_single_nonedge_cell_image_dataset",
    registry="s3://allencell",
    dest=r"\hipsc_single_nonedge_cell"
)

q3.Package.install("aics/hipsc_single_edge_cell_image_dataset",
                   registry="s3://allencell",
                   dest=r"\hipsc_single_edge_cell")


q3.Package.install("aics/hipsc_single_i1_cell_image_dataset",
                   registry="s3://allencell",
                   dest=r"\hipsc_single_i1_cell")

q3.Package.install("aics/hipsc_single_m1_cell_image_dataset",
                   registry="s3://allencell",
                   dest=r"\hipsc_single_m1_cell")


q3.Package.install("aics/hipsc_single_i2_cell_image_dataset",
                   registry="s3://allencell",
                   dest=r"\hipsc_single_i2_cell")

q3.Package.install("aics/hipsc_single_m2_cell_image_dataset",
                   registry="s3://allencell",
                   dest=r"\hipsc_single_m2_cell")