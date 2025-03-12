import math
import random
import numpy as np
import scipy as sp
import autograd.numpy as anp
import pymanopt

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

def normalize_path_length(data: np.ndarray) -> np.ndarray:
  """
  Normalize each row of the data matrix so that the sum of the lengths equals 1
  """
  pathlen = np.linalg.norm(data, axis=1).sum()
  data = data / pathlen
  pathlen2 = np.linalg.norm(data, axis=1).sum()
  return data

def _summed_dir(dat: np.ndarray) -> np.ndarray:
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
    dir_as_col = dir[:,None]
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

def summed_dir_scipyopt(dat: np.ndarray) -> np.ndarray:
  """
  Computes the 'summed directions' vector.
  """

  def objctv(p):
    sum = 0
    for i in range(dat.shape[0]):
      xi = dat[i,:]
      sum += np.linalg.norm(np.dot(p,xi) * xi - xi)**2
    return sum


  def objective(p):
    projections = dat @ p[:,None]
    scaled = dat * projections
    remainders = scaled - dat
    return (remainders * remainders).sum()

  def jac(p):
    projections = dat @ p[:, None]
    diffs = projections*dat - dat
    scalings = (diffs * dat).sum(axis=1)*2
    return (dat*scalings[:,None]).sum(axis=0)

  def grad(p):
    sum = p*0
    for i in range(dat.shape[0]):
      xi = dat[i, :]
      sum += np.dot(np.dot(p, xi) * xi - xi, xi)*2 * xi
    return sum

  def grad2(p):
    sum = p*0
    for i in range(dat.shape[0]):
      xi = dat[i, :]
      pTxi = np.dot(p, xi)
      xiTxi = np.dot(xi, xi)
      #sum += (pTxi*xiTxi*xi - xiTxi*xi)*2
      sum += xiTxi * (pTxi * xi - xi) * 2
    return sum

  x0 = normalize_vec(np.random.rand(dat.shape[1]))
  #print(f"diff obective {objctv(x0) - objective(x0)}")
  #print(f"diff jac {grad(x0) - grad2(x0)}")

  check = sp.optimize.check_grad(objective, jac, x0=x0)
  #print(check)

  def constr(p):
    return (p*p).sum()

  def d_constr(p):
    return 2*p

  nlc = sp.optimize.NonlinearConstraint(constr, lb=1.0, ub=1.0, jac = d_constr)

  res = sp.optimize.minimize(objective, x0=x0, method='trust-constr', jac=jac, constraints=nlc)
  print(res)
  return res.x


  # # calculate square norms for each row of dat
  # norms2 = (dat**2).sum(axis=1)
  # # scale each row of dat by respective squared norm
  # scaled = (dat * norms2[:,None])
  # # sum all rows up
  # dir = scaled.sum(axis=0)
  # return normalize_vec(dir))

def summed_dir(dat: np.ndarray) -> np.ndarray:
  """
  Computes the 'summed directions' vector.
  """
  dim = dat.shape[1]
  manifold = pymanopt.manifolds.Sphere(dim)

  @pymanopt.function.autograd(manifold)
  def objective(p):
    projections = dat @ p[:,None]
    scaled = dat * projections
    remainders = scaled - dat
    return (remainders * remainders).sum()

  @pymanopt.function.autograd(manifold)
  def jac(p):
    projections = dat @ p[:, None]
    diffs = projections*dat - dat
    scalings = (diffs * dat).sum(axis=1)*2
    return (dat*scalings[:,None]).sum(axis=0)

  problem = pymanopt.Problem(manifold=manifold, cost=objective, euclidean_gradient=jac)
  optimizer = pymanopt.optimizers.SteepestDescent()
  result = optimizer.run(problem)

  return result.point

def test1():
  dat = np.random.rand(7,3) *2 -1
  dat = np.hstack([dat,dat*1.2 +.1])
  # mu
  mu = (dat**2).sum(axis=1)[:,None] * dat
  mu = mu.sum(axis=0)
  # cov
  cov = dat.T @ dat
  lam = 4.44  #random.random()
  reg = lam*np.eye(cov.shape[0])
  sum_inv = np.linalg.inv(cov + reg)
  inv_sum = np.linalg.inv(cov) + np.linalg.inv(reg)
  print(np.linalg.norm(sum_inv @ mu, ord=2))
  print(np.linalg.norm(mu, ord=2))
  print(np.linalg.norm(sum_inv, ord=2)*np.linalg.norm(mu, ord=2))

def test2():
  dat = np.random.rand(7, 3) * 2 - 1
  dat = np.hstack([dat, dat * 1.2 + .1])
  # mu
  #mu = (dat ** 2).sum(axis=1)[:, None] * dat
  mu = dat.sum(axis=0)
  cov = dat.T @ dat







  


if __name__ == '__main__':
  test2()
  #data = np.random.rand(3, 2) * 10 - 5
  #summed_dir_2(data)
