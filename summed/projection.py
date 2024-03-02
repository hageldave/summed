import numpy as np

def normalize_vec(vec: np.ndarray) -> np.ndarray:
  """
  Returns the vector normalized to unit length.
  In case the vector is close to zero, the same vector is returned.
  """
  norm = np.linalg.norm(vec)
  if norm < 1e-10:
    return vec
  else:
    return vec * (1/norm)


def normalize_rows(data: np.ndarray) -> np.ndarray:
  """
  Normalizes each row of the data matrix to be of unit length.
  """
  return np.array([
      normalize_vec(data[i]) for i in range(data.shape[0])
  ])


def summed_dir(dat: np.ndarray) -> np.ndarray:
  """
  Computes the 'summed directions' vector.
  """
  # calculate square norms for each row of dat
  norms2 = (dat**2).sum(axis=1)
  # scale each row of dat by respective squared norm
  scaled = (dat * norms2[:,None])
  # sum all rows up
  dir = scaled.sum(axis=0)
  return normalize_vec(dir)


def summed_dirs(dat: np.ndarray, numDirs) -> np.ndarray:
  """
  Computes a set of 'summed directions' vectors.
  For the (i+1)th vector the ith vector's direction is removed from the data
  and then the remaining data is used to compute the sum.
  """
  dirs = np.array([])
  for i in range(0, numDirs):
    dir = summed_dir(dat)
    dir_as_col = dir[:,None];
    dirs = np.hstack((dirs, dir_as_col)) if dirs.size else dir_as_col
    scale = dat @ dir_as_col
    dat = dat-(scale * dir)
  return dirs


def summed_dir_projection(dat: np.ndarray) -> np.ndarray:
  # center data
  #dat = dat - dat.mean(axis=0)
  proj_mat = summed_dirs(dat, 2)
  data_proj = dat @ proj_mat
  return data_proj

def summed_dir_projection_and_transform(dat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  # center data
  #dat = dat - dat.mean(axis=0)
  proj_mat = summed_dirs(dat, 2)
  data_proj = dat @ proj_mat
  return data_proj, proj_mat

