#!/usr/bin/env python3
"""Controls for the D4 scalar symbol and site-reflection factorization."""

import argparse
from fractions import Fraction as F
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg, special

ROOTS = tuple(tuple(sign[ij.index(i)] if i in ij else 0 for i in range(4))
              for ij in combinations(range(4), 2) for sign in product((-1,1), repeat=2))


def moment(k, factor, n):
    return factor**(n//2)*sum(sum(v_i*k_i for v_i,k_i in zip(v,k))**n for v in ROOTS)


def symbol(a, k):
    theta = a/np.sqrt(2)*np.array(ROOTS).dot(k)
    return 2*np.sum(np.sin(theta/2)**2)/(3*a*a)


def symbol_check():
    assert len(ROOTS) == len(set(ROOTS)) == 24
    c = {}
    for k, factor, p6 in [((1,0,0,0), F(1,2), F(3,2)),
                           ((1,1,0,0), F(1,4), F(9,4))]:
        assert moment(k, factor, 2) == 6 and moment(k, factor, 4) == 3
        assert moment(k, factor, 6) == p6
    diag, axis = np.array([1.,1.,0.,0.])/np.sqrt(2), np.array([1.,0.,0.,0.])
    assert symbol(.5,np.zeros(4)) == 0
    for a in (.5,.25,.125):
        gap = (symbol(a,diag)-symbol(a,axis))/a**4
        assert abs(gap-1/2880) <= 2*a*a/5040+2e-12
        for k in (axis,diag):
            p6 = 1.5 if k is axis else 2.25
            assert abs(symbol(a,k)-(1-a*a/24+a**4*p6/2160)) <= a**6/5040+1e-14
        c[str(a)] = gap
    wrong = lambda h: (2*(1-math.cos(h)) - 4*(1-math.cos(h/np.sqrt(2))))/h**4
    assert abs(wrong(.125)+1/24) < 1e-4  # axial hypercubic stencil: r=4 anisotropy
    return c


def gaussian_check(radius, mass=1.0):
    sites = sorted(n for n in product(range(-radius,radius+1), repeat=4) if sum(n)%2 == 0)
    loc = {r:i for i,r in enumerate(sites)}
    S = np.eye(len(sites))*(mass**2+8)
    for j,n in enumerate(sites):
        for v in ROOTS:
            neighbor = tuple(ni+vi for ni,vi in zip(n,v))
            if neighbor in loc:
                S[j,loc[neighbor]] = -1/3
    pos = [i for i,n in enumerate(sites) if n[0]>0]
    neg = [loc[(-sites[i][0], *sites[i][1:])] for i in pos]
    zero = [i for i,n in enumerate(sites) if n[0]==0]
    assert not S[np.ix_(pos,neg)].any()
    assert np.array_equal(S[np.ix_(pos,zero)],S[np.ix_(neg,zero)])
    assert np.array_equal(S[np.ix_(pos,pos)],S[np.ix_(neg,neg)])
    pi,ref = np.meshgrid(pos,neg,indexing='ij')
    C = linalg.cho_solve(linalg.cho_factor(S),np.eye(len(S)))
    cross = C[pi,ref]
    assert np.max(abs(cross-cross.T)) < 2e-15
    Ah = S[np.ix_(pos,pos)]
    H = S[np.ix_(pos,zero)]
    projected = linalg.solve(Ah,H,assume_a="pos")
    pred = np.einsum("is,js->ij",
                     np.einsum("it,ts->is",projected,C[np.ix_(zero,zero)]),projected)
    np.testing.assert_allclose(cross,pred,atol=2e-15,rtol=1e-12)
    lo = float(linalg.eigvalsh(cross)[0])
    assert lo > -2e-15
    return {"sites": len(sites), "mass": mass, "min_reflected_eigenvalue": lo,
            "factorization_residual": float(np.max(abs(cross-pred)))}


def quartic_three_site():
    # Independent quadrature of the reflected functional and conditional norm.
    x,w = special.roots_hermite(48)
    w = w*np.exp(-.1*x**4)
    bound = np.exp(-.5*(x[:,None]-x[None,:])**2)
    f = x**3+1j*x**2
    z = np.sum(w*np.einsum("ki,i->k",bound,w)**2)
    full = np.einsum("k,ki,kj,i,j,i,j->",w,bound,bound,w,w,f,f.conj())/z
    cond = np.sum(w*abs(np.einsum("ki,i->k",bound,w*f))**2)/z
    assert cond > 0 and abs(full-cond) < 1e-13
    return float(cond)


def run():
    # Reflection symmetric and strictly stable, yet RP fails with a positive
    # off-diagonal precision entry across the slice: C_{+,-}=-1/8.
    bad = np.array([[3.,1.],[1.,3.]])
    assert linalg.eigvalsh(bad)[0] == 2
    cross_bad = float(linalg.inv(bad)[0,1])
    assert abs(cross_bad+1/8) < 1e-14
    return {"scope":"scalar site reflection only; no interacting gauge or chiral theorem",
            "numpy":np.__version__, "reduced_angular_gap":symbol_check(),
            "gap_limit":1/2880, "stable_bad_bond_reflected_covariance":cross_bad,
            "gaussian_dirichlet":[gaussian_check(1,.5),gaussian_check(2,1.)],
            "quartic_three_site_reflected_norm":quartic_three_site()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = run()
    if a.output is not None:
        with a.output.open("x",encoding="utf-8") as handle:
            json.dump(result,handle,indent=2,allow_nan=False)
            handle.write("\n")
    print("D4_SCALAR_SITE_REFLECTION_PASS",json.dumps(result,allow_nan=False))
