from pathlib import Path

import h5py


path = Path("signals.h5") # Change it to where your h5 file lives

with h5py.File(path, "r") as f:
    print("Events:", f.attrs["n_written"])
    print("Complete:", bool(f.attrs["complete"]))

    # Binary parameters
    mass_1 = f["parameters/mass_1"][:]

    # Shared GW frequency grid and the first event's polarisations
    frequencies = f["gw/frequencies"][:]
    hp = f["gw/p"][0]
    hc = f["gw/c"][0]

    # First event's ztfg light curve
    time = f["em/ztfg/time"][0]
    mag = f["em/ztfg/mag"][0]

print("First mass_1:", mass_1[0])
print("GW samples:", hp.shape)
print("ztfg samples:", mag.shape)

# If you wish to inspect the structure of the h5 file,
# Uncomment the following:

# def show_structure(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
#     else:
#         print(f"{name}/")
# 
# 
# with h5py.File(path, "r") as f:
#     print("HDF5 structure:")
#     f.visititems(show_structure)